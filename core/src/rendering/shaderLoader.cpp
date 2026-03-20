#include <core/backgroundTask.h>
#include <core/diagnostics.h>
#include <core/rendering/shaderLoader.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <string>
#include <unordered_map>

#ifdef PTS_SHADER_HOT_RELOAD
#include <slang-com-ptr.h>
#include <slang.h>

#include <thread>
#endif

using namespace pts::rendering;

// ---------------------------------------------------------------------------
// SlangCompiler (hot-reload only)
// ---------------------------------------------------------------------------

#ifdef PTS_SHADER_HOT_RELOAD

class SlangCompiler {
   public:
    struct CompileResult {
        bool success = false;
        std::vector<std::string> wgsl;
        std::vector<std::filesystem::path> dependencies;
        std::string diagnostics_text;
    };

    SlangCompiler(std::filesystem::path search_path, std::shared_ptr<spdlog::logger> logger);
    ~SlangCompiler();

    CompileResult compile(const std::filesystem::path& slang_source,
                          const std::vector<std::string>& entry_points);

   private:
    std::filesystem::path m_search_path;
    std::shared_ptr<spdlog::logger> m_logger;
};

SlangCompiler::SlangCompiler(std::filesystem::path search_path,
                             std::shared_ptr<spdlog::logger> logger)
    : m_search_path(std::move(search_path)), m_logger(std::move(logger)) {
}

SlangCompiler::~SlangCompiler() = default;

SlangCompiler::CompileResult SlangCompiler::compile(const std::filesystem::path& slang_source,
                                                    const std::vector<std::string>& entry_points) {
    CompileResult result;

    // Fresh global session each call — Slang caches loaded modules by name,
    // so reusing a session returns stale code after the source file changes.
    Slang::ComPtr<slang::IGlobalSession> global_session;
    auto hr = slang::createGlobalSession(global_session.writeRef());
    if (SLANG_FAILED(hr) || !global_session) {
        result.diagnostics_text = "Failed to create Slang global session";
        return result;
    }

    slang::SessionDesc session_desc = {};
    slang::TargetDesc target_desc = {};
    target_desc.format = SLANG_WGSL;
    session_desc.targets = &target_desc;
    session_desc.targetCount = 1;
    // Match CLI slangc default: column-major matrix layout
    session_desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;

    auto search_str = m_search_path.string();
    auto source_dir_str = slang_source.parent_path().string();
    const char* search_paths[] = {source_dir_str.c_str(), search_str.c_str()};
    session_desc.searchPaths = search_paths;
    session_desc.searchPathCount = 2;

    Slang::ComPtr<slang::ISession> session;
    hr = global_session->createSession(session_desc, session.writeRef());
    if (SLANG_FAILED(hr) || !session) {
        result.diagnostics_text = "Failed to create Slang session";
        return result;
    }

    auto module_name = slang_source.stem().string();
    Slang::ComPtr<slang::IBlob> diagnostics;
    auto* module = session->loadModule(module_name.c_str(), diagnostics.writeRef());
    if (diagnostics) {
        result.diagnostics_text = static_cast<const char*>(diagnostics->getBufferPointer());
    }
    if (!module) {
        return result;
    }

    auto dep_count = module->getDependencyFileCount();
    for (SlangInt32 i = 0; i < dep_count; ++i) {
        auto* dep_path = module->getDependencyFilePath(i);
        if (dep_path) {
            result.dependencies.emplace_back(dep_path);
        }
    }

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

    Slang::ComPtr<slang::IComponentType> linked;
    hr = program->link(linked.writeRef(), diagnostics.writeRef());
    if (diagnostics) {
        result.diagnostics_text += static_cast<const char*>(diagnostics->getBufferPointer());
    }
    if (SLANG_FAILED(hr) || !linked) {
        return result;
    }

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
// ShaderLoader::Impl
// ---------------------------------------------------------------------------

struct ShaderLoader::Impl {
    struct ShaderEntry {
        std::string resource_key;
        std::string slang_source;
        std::string wgsl_output;
        EmbeddedGetter embedded_getter;
        std::string cached_wgsl;
#ifdef PTS_SHADER_HOT_RELOAD
        std::vector<std::string> entry_points;
        std::vector<std::pair<std::filesystem::path, std::filesystem::file_time_type>> dependencies;
#endif
    };

#ifdef PTS_SHADER_HOT_RELOAD
    struct ReloadResult {
        struct ShaderResult {
            std::string resource_key;
            std::vector<std::string> wgsl;
            std::vector<std::filesystem::path> dependencies;
            bool success = false;
            std::string diagnostics;
        };
        std::vector<ShaderResult> results;
    };
    std::unique_ptr<pts::BackgroundTask<ReloadResult>> reload_task;
    std::unique_ptr<SlangCompiler> compiler;
#endif

    std::unordered_map<std::string, ShaderEntry> entries;
    std::shared_ptr<spdlog::logger> logger;
};

// ---------------------------------------------------------------------------
// ShaderLoader
// ---------------------------------------------------------------------------

ShaderLoader::ShaderLoader(std::shared_ptr<spdlog::logger> logger)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->logger = std::move(logger);
}

ShaderLoader::~ShaderLoader() = default;
ShaderLoader::ShaderLoader(ShaderLoader&&) noexcept = default;
ShaderLoader& ShaderLoader::operator=(ShaderLoader&&) noexcept = default;

void ShaderLoader::register_shader(std::string_view resource_key, std::string_view slang_source,
                                   std::string_view wgsl_output, EmbeddedGetter embedded_getter,
                                   std::vector<std::string> entry_points) {
    PRECONDITION_MSG(embedded_getter, "embedded_getter must not be null");
    auto key = std::string(resource_key);
    auto embedded = embedded_getter(resource_key);
    PRECONDITION_MSG(embedded.has_value(), "embedded resource must exist at registration time");
    Impl::ShaderEntry entry;
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

    if (fs::is_regular_file(slang_path)) {
        if (!m_impl->compiler) {
            m_impl->compiler = std::make_unique<SlangCompiler>(workspace_root / "core" / "shaders",
                                                               m_impl->logger);
        }

        auto compile_result = m_impl->compiler->compile(slang_path, entry.entry_points);
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
                m_impl->logger->warn("Initial slang compile for {}: {}", slang_source,
                                     compile_result.diagnostics_text);
            }
        }
    } else {
        entry.dependencies.emplace_back(slang_path, fs::file_time_type{});
    }
#endif
    m_impl->entries.emplace(std::move(key), std::move(entry));
}

auto ShaderLoader::load(std::string_view resource_key) const -> std::string {
    auto it = m_impl->entries.find(std::string(resource_key));
    PRECONDITION_MSG(it != m_impl->entries.end(), "Unknown shader resource_key");
    return it->second.cached_wgsl;
}

bool ShaderLoader::poll_and_start_reload() {
#ifdef PTS_SHADER_HOT_RELOAD
    if (m_impl->reload_task) return false;

    namespace fs = std::filesystem;

    std::vector<std::string> dirty_keys;
    for (auto& [key, entry] : m_impl->entries) {
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

    m_impl->logger->info("Shader change detected, recompiling {} shader(s) via libslang",
                         dirty_keys.size());

    struct CompileJob {
        std::string resource_key;
        fs::path slang_source;
        std::vector<std::string> entry_points;
    };
    std::vector<CompileJob> jobs;
    fs::path workspace_root(PTS_WORKSPACE_ROOT);
    for (const auto& key : dirty_keys) {
        auto& entry = m_impl->entries.at(key);
        jobs.push_back({key, workspace_root / entry.slang_source, entry.entry_points});
    }

    auto* compiler = m_impl->compiler.get();
    m_impl->reload_task = std::make_unique<pts::BackgroundTask<Impl::ReloadResult>>(
        "Compiling Shaders",
        [compiler, jobs = std::move(jobs)](pts::TaskProgress& progress) -> Impl::ReloadResult {
            Impl::ReloadResult result;
            for (size_t i = 0; i < jobs.size(); ++i) {
                progress.set_progress(static_cast<float>(i) / static_cast<float>(jobs.size()));
                progress.set_status("Compiling " + jobs[i].resource_key);

                auto cr = compiler->compile(jobs[i].slang_source, jobs[i].entry_points);
                Impl::ReloadResult::ShaderResult sr;
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
    return m_impl->reload_task && !m_impl->reload_task->is_done();
#else
    return false;
#endif
}

auto ShaderLoader::try_finish_reload() -> std::vector<std::string> {
#ifdef PTS_SHADER_HOT_RELOAD
    if (!m_impl->reload_task || !m_impl->reload_task->is_done()) return {};

    auto reload_result = m_impl->reload_task->take_result();
    m_impl->reload_task.reset();

    namespace fs = std::filesystem;
    std::vector<std::string> changed;

    for (auto& sr : reload_result.results) {
        auto it = m_impl->entries.find(sr.resource_key);
        INVARIANT_MSG(it != m_impl->entries.end(), "Reload result for unknown shader key");
        auto& entry = it->second;

        if (!sr.success) {
            m_impl->logger->error("Shader recompilation failed for {}: {}", sr.resource_key,
                                  sr.diagnostics);
            for (auto& [dep_path, last_mtime] : entry.dependencies) {
                std::error_code ec;
                last_mtime = fs::last_write_time(dep_path, ec);
            }
            continue;
        }

        INVARIANT_MSG(!sr.wgsl.empty(), "Successful compile must produce WGSL output");
        auto& new_wgsl = sr.wgsl[0];

        if (new_wgsl != entry.cached_wgsl) {
            entry.cached_wgsl = std::move(new_wgsl);
            changed.push_back(sr.resource_key);
        }

        entry.dependencies.clear();
        for (const auto& dep : sr.dependencies) {
            std::error_code ec;
            auto mtime = fs::last_write_time(dep, ec);
            entry.dependencies.emplace_back(dep, ec ? fs::file_time_type{} : mtime);
        }

        if (!sr.diagnostics.empty()) {
            m_impl->logger->warn("Shader {} diagnostics: {}", sr.resource_key, sr.diagnostics);
        }
    }

    if (!changed.empty()) {
        m_impl->logger->info("Reloaded {} shader(s) via libslang", changed.size());
    }
    return changed;
#else
    return {};
#endif
}

auto ShaderLoader::poll_and_reload() -> std::vector<std::string> {
#ifdef PTS_SHADER_HOT_RELOAD
    poll_and_start_reload();
    if (m_impl->reload_task) {
        while (!m_impl->reload_task->is_done()) {
            std::this_thread::yield();
        }
        return try_finish_reload();
    }
    return {};
#else
    return {};
#endif
}
