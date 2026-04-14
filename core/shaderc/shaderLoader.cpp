#include <core/diagnostics.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <spdlog/spdlog.h>

#include <string>
#include <utility>

namespace pts::rendering {

ShaderLoader::ShaderLoader(std::shared_ptr<spdlog::logger> logger) : m_logger(std::move(logger)) {
}

ShaderLoader::~ShaderLoader() = default;
ShaderLoader::ShaderLoader(ShaderLoader&&) noexcept = default;
ShaderLoader& ShaderLoader::operator=(ShaderLoader&&) noexcept = default;

auto ShaderLoader::logger() const -> const std::shared_ptr<spdlog::logger>& {
    return m_logger;
}

void ShaderLoader::register_shader(std::string_view resource_key, std::string_view slang_source,
                                   std::string_view wgsl_output, EmbeddedGetter embedded_getter,
                                   std::vector<std::string> entry_points) {
    PRECONDITION_MSG(embedded_getter, "embedded_getter must not be null");
    // Embedded WGSL may be absent at registration time on native builds: with
    // Step 6 we only embed WGSL on Emscripten, since native always routes
    // through SlangCompiler. load() is the call site that still demands a hit,
    // and it will panic loudly if the key is ever reached without an embed.
    Entry entry;
    entry.resource_key = std::string(resource_key);
    entry.slang_source = std::string(slang_source);
    entry.wgsl_output = std::string(wgsl_output);
    entry.embedded_getter = embedded_getter;
    entry.entry_points = std::move(entry_points);
    m_entries.emplace(std::string(resource_key), std::move(entry));
}

auto ShaderLoader::load(std::string_view resource_key) const -> std::string {
    auto it = m_entries.find(std::string(resource_key));
    if (it != m_entries.end()) {
        auto embedded = it->second.embedded_getter(resource_key);
        PRECONDITION_MSG(embedded.has_value(), "embedded resource missing for registered key");
        return std::string(*embedded);
    }
    // Not directly registered -- may be a derived variant key (e.g. NO_DEBUG).
    // Probe every registered entry's embedded_getter; first hit wins.
    for (const auto& [_, entry] : m_entries) {
        auto embedded = entry.embedded_getter(resource_key);
        if (embedded.has_value()) {
            return std::string(*embedded);
        }
    }
    PANIC("Unknown shader resource_key");
}

auto ShaderLoader::find(std::string_view resource_key) const noexcept -> const Entry* {
    auto it = m_entries.find(std::string(resource_key));
    return it == m_entries.end() ? nullptr : &it->second;
}

}  // namespace pts::rendering
