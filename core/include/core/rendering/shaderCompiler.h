#pragma once

#include <boost/container_hash/hash.hpp>
#include <boost/core/span.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pts::rendering {

class ShaderLoader;

/// Identifies a shader variant to compile or look up.
///
/// Fields may be extended with additional variant axes (PSO config, material
/// features, vertex layout, ...) without churning IShaderCompiler call sites.
/// Defines must be in a canonical (sorted) order for `operator==` / `hash_value`
/// to agree across semantically equal sets -- callers typically pull these from
/// `shader_variants_map` or other deterministic sources.
struct ShaderKey {
    std::string_view source;
    boost::span<const std::string_view> defines{};
};

inline bool operator==(const ShaderKey& a, const ShaderKey& b) noexcept {
    if (a.source != b.source) return false;
    if (a.defines.size() != b.defines.size()) return false;
    for (std::size_t i = 0; i < a.defines.size(); ++i) {
        if (a.defines[i] != b.defines[i]) return false;
    }
    return true;
}

inline bool operator!=(const ShaderKey& a, const ShaderKey& b) noexcept {
    return !(a == b);
}

inline std::size_t hash_value(const ShaderKey& k) noexcept {
    std::size_t h = 0;
    boost::hash_combine(h, k.source);
    for (const auto& d : k.defines) {
        boost::hash_combine(h, d);
    }
    return h;
}

/// Polymorphic shader-compilation interface.
///
/// FrameGraph uses this to obtain WGSL source for a shader module, keyed by a
/// ShaderKey (source resource key + preprocessor defines + future variant
/// axes). Backends:
///  - EmbeddedCompiler: serves pre-built embedded WGSL (used on WASM and as a
///    native error-fallback).
///  - SlangCompiler: invokes libslang directly, with on-disk caching and
///    mtime-based change detection.
class IShaderCompiler {
   public:
    virtual ~IShaderCompiler() = default;

    /// Return WGSL source for `key`. Fails loud if the key is unknown to this
    /// backend.
    virtual std::string compile(const ShaderKey& key) = 0;

    /// Poll for any sources whose on-disk content has changed since the last
    /// poll. Returns the list of source keys that need to be invalidated.
    /// Default: no-op (returns empty), suitable for static/embedded backends.
    virtual std::vector<std::string> poll_dirty() {
        return {};
    }

    /// Monotonic revision counter for a source key. FrameGraph uses this as a
    /// DepTrackedSlotMap dep -- when the revision changes, dependent shader
    /// modules and pipelines are rebuilt. Tracked per-source (not per-variant)
    /// since every variant of a source rebuilds together.
    [[nodiscard]] virtual uint64_t source_revision(std::string_view source_key) const = 0;

    /// Mark a source dirty. Default no-op for backends with static sources.
    virtual void invalidate(std::string_view /*source_key*/) {
    }
};

/// Thin IShaderCompiler that serves pre-embedded WGSL.
class EmbeddedCompiler final : public IShaderCompiler {
   public:
    explicit EmbeddedCompiler(const ShaderLoader& loader) noexcept;

    std::string compile(const ShaderKey& key) override;

    [[nodiscard]] uint64_t source_revision(std::string_view source_key) const override;

    void invalidate(std::string_view source_key) override;

   private:
    const ShaderLoader* m_loader;
    // Revision-per-source-key. Bumped by invalidate(). Defaults to 1 on first
    // read. On pure-embedded platforms nothing ever calls invalidate(), so
    // source_revision() stays constant.
    std::unordered_map<std::string, uint64_t> m_revisions;
};

/// Build the shader compiler for the current platform.
/// - Native: `SlangCompiler` primary, `EmbeddedCompiler` as error-fallback.
/// - WASM:   `EmbeddedCompiler` only.
std::unique_ptr<IShaderCompiler> make_shader_compiler(const ShaderLoader& loader);

}  // namespace pts::rendering
