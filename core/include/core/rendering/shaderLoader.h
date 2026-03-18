#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pts {
template <typename T>
class BackgroundTask;
}

namespace spdlog {
class logger;
}

namespace pts::rendering {

/// Function pointer matching the generated get_resource() signature.
using EmbeddedGetter = std::optional<std::string_view> (*)(std::string_view);

class ShaderLoader {
   public:
    explicit ShaderLoader(std::shared_ptr<spdlog::logger> logger);
    ~ShaderLoader();

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
    /// Always returns the last successfully loaded source (seeded from embedded on register).
    /// After a successful poll_and_reload, returns the reloaded version.
    /// After a failed recompilation, keeps returning the last-good version.
    [[nodiscard]] auto load(std::string_view resource_key) const -> std::string;

    /// Poll .slang source mtimes. If any changed, recompile via slangc subprocess
    /// and re-read .wgsl outputs from disk.
    /// Returns list of resource_keys whose content changed (empty if nothing changed).
    /// No-op in non-hot-reload builds (returns empty).
    [[nodiscard]] auto poll_and_reload() -> std::vector<std::string>;

    /// Start background recompilation if any .slang sources are dirty.
    /// No-op if already reloading or nothing changed.
    /// Returns true if a compilation was started.
    bool poll_and_start_reload();

    /// True if a background compilation is in progress.
    bool is_reloading() const;

    /// If background compilation finished, read .wgsl files and update cache.
    /// Returns list of changed resource keys, or empty if not done yet / nothing changed.
    std::vector<std::string> try_finish_reload();

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
#ifdef PTS_SHADER_HOT_RELOAD
    std::unique_ptr<pts::BackgroundTask<int>> m_reload_task;
#endif
};

}  // namespace pts::rendering
