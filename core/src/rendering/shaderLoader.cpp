#include <core/diagnostics.h>
#include <core/rendering/shaderLoader.h>
#include <spdlog/spdlog.h>

#ifdef PTS_SHADER_HOT_RELOAD
#include <core/backgroundTask.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <thread>
#endif

using namespace pts::rendering;

ShaderLoader::ShaderLoader(std::shared_ptr<spdlog::logger> logger) : m_logger(std::move(logger)) {
}

ShaderLoader::~ShaderLoader() = default;

void ShaderLoader::register_shader(std::string_view resource_key, std::string_view slang_source,
                                   std::string_view wgsl_output, EmbeddedGetter embedded_getter) {
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
    // Seed mtime so the first poll_and_reload() doesn't see every shader as dirty.
    namespace fs = std::filesystem;
    fs::path workspace_root(PTS_WORKSPACE_ROOT);
    std::error_code ec;
    entry.last_mtime = fs::last_write_time(workspace_root / entry.slang_source, ec);
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
    // Task exists (running or done-but-unconsumed) — wait for try_finish_reload().
    if (m_reload_task) return false;

    namespace fs = std::filesystem;
    fs::path workspace_root(PTS_WORKSPACE_ROOT);
    bool any_dirty = false;

    for (auto& [key, entry] : m_entries) {
        auto slang_path = workspace_root / entry.slang_source;
        std::error_code ec;
        auto mtime = fs::last_write_time(slang_path, ec);
        if (ec) {
            m_logger->warn("Cannot stat {}: {}", slang_path.string(), ec.message());
            continue;
        }
        if (mtime != entry.last_mtime) {
            any_dirty = true;
            break;
        }
    }

    if (!any_dirty) return false;

    auto repo_cmd = (workspace_root / "repo").string() + " slangc --force";
    m_logger->info("Shader change detected, recompiling: {}", repo_cmd);

    m_reload_task = std::make_unique<pts::BackgroundTask<int>>(
        "Compiling Shaders", [cmd = repo_cmd](pts::TaskProgress& progress) -> int {
            progress.set_status("Compiling shaders...");
            progress.set_progress(0.5f);
            int ret = std::system(cmd.c_str());
            progress.set_progress(1.0f);
            return ret;
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

    int ret = m_reload_task->take_result();
    m_reload_task.reset();

    namespace fs = std::filesystem;
    fs::path workspace_root(PTS_WORKSPACE_ROOT);

    if (ret != 0) {
        m_logger->error("Shader recompilation failed (exit code {}), keeping last-good shaders",
                        ret);
        for (auto& [key, entry] : m_entries) {
            auto slang_path = workspace_root / entry.slang_source;
            std::error_code ec;
            entry.last_mtime = fs::last_write_time(slang_path, ec);
        }
        return {};
    }

    std::vector<std::string> changed;
    for (auto& [key, entry] : m_entries) {
        auto slang_path = workspace_root / entry.slang_source;
        std::error_code ec;
        entry.last_mtime = fs::last_write_time(slang_path, ec);

        auto wgsl_path = workspace_root / entry.wgsl_output;
        std::ifstream file(wgsl_path, std::ios::binary);
        if (!file) {
            m_logger->error("Failed to read recompiled shader: {}", wgsl_path.string());
            continue;
        }
        std::ostringstream ss;
        ss << file.rdbuf();
        auto new_wgsl = ss.str();

        if (new_wgsl != entry.cached_wgsl) {
            entry.cached_wgsl = std::move(new_wgsl);
            changed.push_back(key);
        }
    }

    if (!changed.empty()) {
        m_logger->info("Reloaded {} shader(s)", changed.size());
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
