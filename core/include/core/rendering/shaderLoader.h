#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace spdlog {
class logger;
}

namespace pts::rendering {

/// Function pointer matching the generated get_resource() signature.
using EmbeddedGetter = std::optional<std::string_view> (*)(std::string_view);

class ShaderLoader {
   public:
    explicit ShaderLoader(std::shared_ptr<spdlog::logger> logger);

    /// Register a shader for loading.
    /// @param resource_key The embedded resource lookup key (e.g.
    /// "editor/generated/shaders/forward.wgsl")
    /// @param slang_source Path to the .slang source file, relative to workspace root (e.g.
    /// "editor/shaders/forward.slang")
    /// @param wgsl_output Path to the compiled .wgsl file, relative to workspace root (e.g.
    /// "editor/generated/shaders/forward.wgsl")
    /// @param embedded_getter Function pointer to the namespace::get_resource function
    void register_shader(std::string_view resource_key, std::string_view slang_source,
                         std::string_view wgsl_output, EmbeddedGetter embedded_getter);

    /// Load shader WGSL source by resource_key.
    /// Hot-reload builds: reads from disk (cached after poll_and_reload).
    /// Non-hot-reload builds: delegates to embedded_getter.
    /// Returns nullopt on failure (caller should keep last-good shader).
    [[nodiscard]] auto load(std::string_view resource_key) const -> std::optional<std::string>;

    /// Poll .slang source mtimes. If any changed, recompile via slangc subprocess
    /// and re-read .wgsl outputs from disk.
    /// Returns list of resource_keys whose content changed (empty if nothing changed).
    /// No-op in non-hot-reload builds (returns empty).
    [[nodiscard]] auto poll_and_reload() -> std::vector<std::string>;

   private:
    struct ShaderEntry {
        std::string resource_key;
        std::string slang_source;  // relative to workspace root
        std::string wgsl_output;   // relative to workspace root
        EmbeddedGetter embedded_getter;
        std::string cached_wgsl;  // last loaded WGSL content (hot-reload only)
#ifdef PTS_SHADER_HOT_RELOAD
        std::filesystem::file_time_type last_mtime{};
#endif
    };

    std::unordered_map<std::string, ShaderEntry> m_entries;
    std::shared_ptr<spdlog::logger> m_logger;
};

}  // namespace pts::rendering
