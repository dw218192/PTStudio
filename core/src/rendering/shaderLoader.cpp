#include <core/rendering/shaderLoader.h>
#include <core/diagnostics.h>
#include <spdlog/spdlog.h>

#ifdef PTS_SHADER_HOT_RELOAD
#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#endif

using namespace pts::rendering;

ShaderLoader::ShaderLoader(std::shared_ptr<spdlog::logger> logger)
    : m_logger(std::move(logger)) {}

void ShaderLoader::register_shader(std::string_view resource_key,
                                    std::string_view slang_source,
                                    std::string_view wgsl_output,
                                    EmbeddedGetter embedded_getter) {
    PRECONDITION_MSG(embedded_getter, "embedded_getter must not be null");
    auto key = std::string(resource_key);
    ShaderEntry entry;
    entry.resource_key = key;
    entry.slang_source = std::string(slang_source);
    entry.wgsl_output = std::string(wgsl_output);
    entry.embedded_getter = embedded_getter;
    m_entries.emplace(std::move(key), std::move(entry));
}

auto ShaderLoader::load(std::string_view resource_key) const -> std::optional<std::string> {
    auto it = m_entries.find(std::string(resource_key));
    PRECONDITION_MSG(it != m_entries.end(), "Unknown shader resource_key");
    auto& entry = it->second;

#ifdef PTS_SHADER_HOT_RELOAD
    if (!entry.cached_wgsl.empty()) {
        return entry.cached_wgsl;
    }
    // Fall through to embedded if no cached version yet
#endif
    auto result = entry.embedded_getter(resource_key);
    if (result.has_value()) {
        return std::optional<std::string>(std::string(*result));
    }
    return std::nullopt;
}

auto ShaderLoader::poll_and_reload() -> std::vector<std::string> {
#ifdef PTS_SHADER_HOT_RELOAD
    namespace fs = std::filesystem;

    // 1. Check mtimes - is anything dirty?
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

    if (!any_dirty) return {};

    // 2. Recompile all shaders via repo slangc
    auto repo_cmd = (workspace_root / "repo").string() + " slangc --force";
    m_logger->info("Shader change detected, recompiling: {}", repo_cmd);

    int ret = std::system(repo_cmd.c_str());
    if (ret != 0) {
        m_logger->error("Shader recompilation failed (exit code {})", ret);
        return {};
    }

    // 3. Re-read .wgsl files and detect changes
    std::vector<std::string> changed;
    for (auto& [key, entry] : m_entries) {
        // Update mtime
        auto slang_path = workspace_root / entry.slang_source;
        std::error_code ec;
        entry.last_mtime = fs::last_write_time(slang_path, ec);

        // Read new wgsl
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
