#include <core/backgroundTask.h>
#include <core/diagnostics.h>
#include <core/rendering/shaderLoader.h>
#include <spdlog/spdlog.h>

#ifdef PTS_SHADER_HOT_RELOAD
#include <slang-com-ptr.h>
#include <slang.h>

#include <array>
#include <thread>
#endif

#ifdef PTS_SHADER_HOT_RELOAD

// ---------------------------------------------------------------------------
// SlangCompiler — defined in .cpp only (PIMPL: slang.h never leaks into the header)
// ---------------------------------------------------------------------------

namespace pts::rendering {

class SlangCompiler {
   public:
    struct CompileResult {
        bool success = false;
        /// WGSL code (single module with all entry points from getTargetCode)
        std::vector<std::string> wgsl;
        /// All file paths involved (entry point + imports)
        std::vector<std::filesystem::path> dependencies;
        std::string diagnostics_text;
    };

    SlangCompiler(std::filesystem::path search_path, std::shared_ptr<spdlog::logger> logger);
    ~SlangCompiler();

    /// Compile a .slang file, returning WGSL for each entry point and the dependency list.
    CompileResult compile(const std::filesystem::path& slang_source,
                          const std::vector<std::string>& entry_points);

   private:
    Slang::ComPtr<slang::IGlobalSession> m_global_session;
    std::filesystem::path m_search_path;
    std::shared_ptr<spdlog::logger> m_logger;
};

}  // namespace pts::rendering

using namespace pts::rendering;

SlangCompiler::SlangCompiler(std::filesystem::path search_path,
                             std::shared_ptr<spdlog::logger> logger)
    : m_search_path(std::move(search_path)), m_logger(std::move(logger)) {
    auto hr = slang::createGlobalSession(m_global_session.writeRef());
    POSTCONDITION_MSG(SLANG_SUCCEEDED(hr) && m_global_session,
                      "Failed to create Slang global session");
}

SlangCompiler::~SlangCompiler() = default;

SlangCompiler::CompileResult SlangCompiler::compile(const std::filesystem::path& slang_source,
                                                    const std::vector<std::string>& entry_points) {
    CompileResult result;

    // Per-compilation session
    slang::SessionDesc session_desc = {};
    slang::TargetDesc target_desc = {};
    target_desc.format = SLANG_WGSL;
    session_desc.targets = &target_desc;
    session_desc.targetCount = 1;

    auto search_str = m_search_path.string();
    const char* search_paths[] = {search_str.c_str()};
    session_desc.searchPaths = search_paths;
    session_desc.searchPathCount = 1;

    Slang::ComPtr<slang::ISession> session;
    auto hr = m_global_session->createSession(session_desc, session.writeRef());
    if (SLANG_FAILED(hr) || !session) {
        result.diagnostics_text = "Failed to create Slang session";
        return result;
    }

    // Load module — module name is the stem (e.g. "forward" from "forward.slang")
    auto module_name = slang_source.stem().string();
    Slang::ComPtr<slang::IBlob> diagnostics;
    auto* module = session->loadModule(module_name.c_str(), diagnostics.writeRef());
    if (diagnostics) {
        result.diagnostics_text = static_cast<const char*>(diagnostics->getBufferPointer());
    }
    if (!module) {
        return result;
    }

    // Collect dependency files
    auto dep_count = module->getDependencyFileCount();
    for (SlangInt32 i = 0; i < dep_count; ++i) {
        auto* dep_path = module->getDependencyFilePath(i);
        if (dep_path) {
            result.dependencies.emplace_back(dep_path);
        }
    }

    // Find entry points
    std::vector<Slang::ComPtr<slang::IEntryPoint>> ep_objects;
    for (const auto& ep_name : entry_points) {
        SlangStage stage = SLANG_STAGE_NONE;
        if (ep_name.find("vs_") == 0 || ep_name.find("vert") == 0) {
            stage = SLANG_STAGE_VERTEX;
        } else if (ep_name.find("fs_") == 0 || ep_name.find("frag") == 0) {
            stage = SLANG_STAGE_FRAGMENT;
        } else if (ep_name.find("cs_") == 0 || ep_name.find("comp") == 0) {
            stage = SLANG_STAGE_COMPUTE;
        }

        Slang::ComPtr<slang::IEntryPoint> ep;
        hr = module->findAndCheckEntryPoint(ep_name.c_str(), stage, ep.writeRef(),
                                            diagnostics.writeRef());
        if (diagnostics) {
            result.diagnostics_text += static_cast<const char*>(diagnostics->getBufferPointer());
        }
        if (SLANG_FAILED(hr) || !ep) {
            m_logger->error("Failed to find entry point '{}' in {}", ep_name,
                            slang_source.string());
            return result;
        }
        ep_objects.push_back(std::move(ep));
    }

    // Create composite component (module + all entry points)
    std::vector<slang::IComponentType*> components;
    components.push_back(module);
    for (auto& ep : ep_objects) {
        components.push_back(ep.get());
    }

    Slang::ComPtr<slang::IComponentType> program;
    hr = session->createCompositeComponentType(components.data(), components.size(),
                                               program.writeRef(), diagnostics.writeRef());
    if (diagnostics) {
        result.diagnostics_text += static_cast<const char*>(diagnostics->getBufferPointer());
    }
    if (SLANG_FAILED(hr) || !program) {
        return result;
    }

    // Link
    Slang::ComPtr<slang::IComponentType> linked;
    hr = program->link(linked.writeRef(), diagnostics.writeRef());
    if (diagnostics) {
        result.diagnostics_text += static_cast<const char*>(diagnostics->getBufferPointer());
    }
    if (SLANG_FAILED(hr) || !linked) {
        return result;
    }

    // Extract single WGSL module with all entry points (matches CLI slangc output)
    Slang::ComPtr<slang::IBlob> code;
    hr = linked->getTargetCode(0, code.writeRef(), diagnostics.writeRef());
    if (diagnostics) {
        result.diagnostics_text += static_cast<const char*>(diagnostics->getBufferPointer());
    }
    if (SLANG_FAILED(hr) || !code) {
        return result;
    }
    result.wgsl.emplace_back(static_cast<const char*>(code->getBufferPointer()),
                             code->getBufferSize());

    result.success = true;
    return result;
}

#endif  // PTS_SHADER_HOT_RELOAD

// ---------------------------------------------------------------------------
// ShaderLoader
// ---------------------------------------------------------------------------

using namespace pts::rendering;

ShaderLoader::ShaderLoader(std::shared_ptr<spdlog::logger> logger) : m_logger(std::move(logger)) {
}

ShaderLoader::~ShaderLoader() = default;

void ShaderLoader::register_shader(std::string_view resource_key, std::string_view slang_source,
                                   std::string_view wgsl_output, EmbeddedGetter embedded_getter,
                                   std::vector<std::string> entry_points) {
    PRECONDITION_MSG(embedded_getter, "embedded_getter must not be null");
    auto key = std::string(resource_key);
    auto embedded = embedded_getter(resource_key);
    PRECONDITION_MSG(embedded.has_value(), "embedded resource must exist at registration time");
    ShaderEntry entry;
    entry.resource_key = key;
    entry.slang_source = std::string(slang_source);
    entry.wgsl_output = std::string(wgsl_output);
    entry.embedded_getter = embedded_getter;
    entry.cached_wgsl = std::string(*embedded);
#ifdef PTS_SHADER_HOT_RELOAD
    entry.entry_points = std::move(entry_points);

    namespace fs = std::filesystem;
    fs::path workspace_root(PTS_WORKSPACE_ROOT);
    auto slang_path = workspace_root / entry.slang_source;

    // Only attempt libslang compilation when the source file exists
    if (fs::is_regular_file(slang_path)) {
        // Lazily create the compiler on first use
        if (!m_compiler) {
            m_compiler =
                std::make_unique<SlangCompiler>(workspace_root / "core" / "shaders", m_logger);
        }

        // Initial compile to discover dependencies
        auto compile_result = m_compiler->compile(slang_path, entry.entry_points);
        if (compile_result.success) {
            for (const auto& dep : compile_result.dependencies) {
                std::error_code ec;
                auto mtime = fs::last_write_time(dep, ec);
                entry.dependencies.emplace_back(dep, ec ? fs::file_time_type{} : mtime);
            }
        } else {
            std::error_code ec;
            auto mtime = fs::last_write_time(slang_path, ec);
            entry.dependencies.emplace_back(slang_path, ec ? fs::file_time_type{} : mtime);
            if (!compile_result.diagnostics_text.empty()) {
                m_logger->warn("Initial slang compile for {}: {}", slang_source,
                               compile_result.diagnostics_text);
            }
        }
    } else {
        // Source doesn't exist yet — track just the entry point path for mtime polling
        entry.dependencies.emplace_back(slang_path, fs::file_time_type{});
    }
#endif
    m_entries.emplace(std::move(key), std::move(entry));
}

auto ShaderLoader::load(std::string_view resource_key) const -> std::string {
    auto it = m_entries.find(std::string(resource_key));
    PRECONDITION_MSG(it != m_entries.end(), "Unknown shader resource_key");
    return it->second.cached_wgsl;
}

bool ShaderLoader::poll_and_start_reload() {
#ifdef PTS_SHADER_HOT_RELOAD
    if (m_reload_task) return false;

    namespace fs = std::filesystem;

    // Collect dirty shaders by checking ALL dependency mtimes
    std::vector<std::string> dirty_keys;
    for (auto& [key, entry] : m_entries) {
        for (auto& [dep_path, last_mtime] : entry.dependencies) {
            std::error_code ec;
            auto mtime = fs::last_write_time(dep_path, ec);
            if (ec) continue;
            if (mtime != last_mtime) {
                dirty_keys.push_back(key);
                break;
            }
        }
    }

    if (dirty_keys.empty()) return false;

    m_logger->info("Shader change detected, recompiling {} shader(s) via libslang",
                   dirty_keys.size());

    // Capture what the background thread needs (no references to m_entries)
    struct CompileJob {
        std::string resource_key;
        fs::path slang_source;
        std::vector<std::string> entry_points;
    };
    std::vector<CompileJob> jobs;
    fs::path workspace_root(PTS_WORKSPACE_ROOT);
    for (const auto& key : dirty_keys) {
        auto& entry = m_entries.at(key);
        jobs.push_back({key, workspace_root / entry.slang_source, entry.entry_points});
    }

    // Capture compiler as raw ptr (it outlives the task)
    auto* compiler = m_compiler.get();
    m_reload_task = std::make_unique<pts::BackgroundTask<ReloadResult>>(
        "Compiling Shaders",
        [compiler, jobs = std::move(jobs)](pts::TaskProgress& progress) -> ReloadResult {
            ReloadResult result;
            for (size_t i = 0; i < jobs.size(); ++i) {
                progress.set_progress(static_cast<float>(i) / static_cast<float>(jobs.size()));
                progress.set_status("Compiling " + jobs[i].resource_key);

                auto cr = compiler->compile(jobs[i].slang_source, jobs[i].entry_points);
                ReloadResult::ShaderResult sr;
                sr.resource_key = jobs[i].resource_key;
                sr.success = cr.success;
                sr.wgsl = std::move(cr.wgsl);
                sr.dependencies = std::move(cr.dependencies);
                sr.diagnostics = std::move(cr.diagnostics_text);
                result.results.push_back(std::move(sr));
            }
            progress.set_progress(1.0f);
            return result;
        });

    return true;
#else
    return false;
#endif
}

bool ShaderLoader::is_reloading() const {
#ifdef PTS_SHADER_HOT_RELOAD
    return m_reload_task && !m_reload_task->is_done();
#else
    return false;
#endif
}

auto ShaderLoader::try_finish_reload() -> std::vector<std::string> {
#ifdef PTS_SHADER_HOT_RELOAD
    if (!m_reload_task || !m_reload_task->is_done()) return {};

    auto reload_result = m_reload_task->take_result();
    m_reload_task.reset();

    namespace fs = std::filesystem;
    std::vector<std::string> changed;

    for (auto& sr : reload_result.results) {
        auto it = m_entries.find(sr.resource_key);
        INVARIANT_MSG(it != m_entries.end(), "Reload result for unknown shader key");
        auto& entry = it->second;

        if (!sr.success) {
            m_logger->error("Shader recompilation failed for {}: {}", sr.resource_key,
                            sr.diagnostics);
            // Update mtimes so we don't re-trigger on every poll
            for (auto& [dep_path, last_mtime] : entry.dependencies) {
                std::error_code ec;
                last_mtime = fs::last_write_time(dep_path, ec);
            }
            continue;
        }

        // getTargetCode produces a single WGSL module with all entry points
        INVARIANT_MSG(!sr.wgsl.empty(), "Successful compile must produce WGSL output");
        auto& new_wgsl = sr.wgsl[0];

        if (new_wgsl != entry.cached_wgsl) {
            entry.cached_wgsl = std::move(new_wgsl);
            changed.push_back(sr.resource_key);
        }

        // Re-discover dependencies (import list may change between compiles)
        entry.dependencies.clear();
        for (const auto& dep : sr.dependencies) {
            std::error_code ec;
            auto mtime = fs::last_write_time(dep, ec);
            entry.dependencies.emplace_back(dep, ec ? fs::file_time_type{} : mtime);
        }

        if (!sr.diagnostics.empty()) {
            m_logger->warn("Shader {} diagnostics: {}", sr.resource_key, sr.diagnostics);
        }
    }

    if (!changed.empty()) {
        m_logger->info("Reloaded {} shader(s) via libslang", changed.size());
    }
    return changed;
#else
    return {};
#endif
}

auto ShaderLoader::poll_and_reload() -> std::vector<std::string> {
#ifdef PTS_SHADER_HOT_RELOAD
    poll_and_start_reload();
    if (m_reload_task) {
        while (!m_reload_task->is_done()) {
            std::this_thread::yield();
        }
        return try_finish_reload();
    }
    return {};
#else
    return {};
#endif
}
