#pragma once

#include <boost/core/span.hpp>
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

/// Registry of shader source metadata (slang path + embedded WGSL fallback +
/// entry points). Does not compile shaders -- `SlangCompiler` / `EmbeddedCompiler`
/// consume this registry and produce WGSL.
///
/// Kept as a thin shim so existing pass/renderer ctors that take a
/// `ShaderLoader&` continue to compile. All libslang wrapper logic and the
/// async hot-reload plumbing have been moved into `SlangCompiler`.
class ShaderLoader {
   public:
    struct Entry {
        std::string resource_key;
        std::string slang_source;  // path relative to workspace root
        std::string wgsl_output;   // pre-compiled embedded variant key
        EmbeddedGetter embedded_getter = nullptr;
        std::vector<std::string> entry_points;
    };

    explicit ShaderLoader(std::shared_ptr<spdlog::logger> logger);
    ~ShaderLoader();

    ShaderLoader(ShaderLoader&&) noexcept;
    ShaderLoader& operator=(ShaderLoader&&) noexcept;

    /// Register a shader's metadata.
    void register_shader(std::string_view resource_key, std::string_view slang_source,
                         std::string_view wgsl_output, EmbeddedGetter embedded_getter,
                         std::vector<std::string> entry_points = {"vs_main", "fs_main"});

    /// Return the embedded WGSL at `resource_key`. Fails loud if the key is
    /// not a registered resource AND the embedded_getter of any registered
    /// entry cannot resolve it either.
    [[nodiscard]] auto load(std::string_view resource_key) const -> std::string;

    /// Lookup a registered entry. Returns nullptr if not registered.
    [[nodiscard]] auto find(std::string_view resource_key) const noexcept -> const Entry*;

    /// Iterate all registered entries.
    template <typename Fn>
    void for_each(Fn&& fn) const {
        for (const auto& [_, entry] : m_entries) fn(entry);
    }

    [[nodiscard]] auto logger() const -> const std::shared_ptr<spdlog::logger>&;

   private:
    std::unordered_map<std::string, Entry> m_entries;
    std::shared_ptr<spdlog::logger> m_logger;
};

}  // namespace pts::rendering
