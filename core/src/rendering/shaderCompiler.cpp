#include <core/diagnostics.h>
#include <core/rendering/shaderCompiler.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <shader_variants_map.h>

#include <algorithm>
#include <string>
#include <vector>

#ifndef __EMSCRIPTEN__
#include <core/rendering/slangCompiler.h>

#include <filesystem>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#endif

namespace pts::rendering {

namespace {

// Sorted + '\n'-terminated join -- must match canonical_defines() in
// slangCompiler.cpp so the generated variant map's defines_canon compares
// byte-for-byte.
std::string canonical_defines_join(boost::span<const std::string_view> defines) {
    std::vector<std::string_view> sorted(defines.begin(), defines.end());
    std::sort(sorted.begin(), sorted.end());
    std::string out;
    for (auto& d : sorted) {
        out.append(d.data(), d.size());
        out.push_back('\n');
    }
    return out;
}

// Map a ShaderKey to the pre-compiled variant resource key using the
// config-driven variant map generated at build time.
std::string derive_variant_key(const ShaderKey& key) {
    auto canon = canonical_defines_join(key.defines);
    for (const auto& v : variants::k_variants) {
        if (v.defines_canon == canon) {
            auto out = std::string(key.source);
            auto dot = out.rfind('.');
            INVARIANT_MSG(dot != std::string::npos, "source_key must have an extension");
            return out.substr(0, dot) + std::string(v.suffix) + out.substr(dot);
        }
    }
    INVARIANT_MSG(false, "EmbeddedCompiler: no registered variant matches the requested defines");
    return {};
}

}  // namespace

EmbeddedCompiler::EmbeddedCompiler(const ShaderLoader& loader) noexcept : m_loader(&loader) {
}

std::string EmbeddedCompiler::compile(const ShaderKey& key) {
    if (key.defines.empty()) {
        return m_loader->load(key.source);
    }
    auto variant_key = derive_variant_key(key);
    return m_loader->load(variant_key);
}

uint64_t EmbeddedCompiler::source_revision(std::string_view source_key) const {
    auto it = m_revisions.find(std::string(source_key));
    return it == m_revisions.end() ? 1 : it->second;
}

void EmbeddedCompiler::invalidate(std::string_view source_key) {
    auto key = std::string(source_key);
    auto it = m_revisions.find(key);
    if (it == m_revisions.end()) {
        m_revisions.emplace(std::move(key), 2);
    } else {
        ++it->second;
    }
}

namespace {

#ifndef __EMSCRIPTEN__
/// Native backend: SlangCompiler only. No embedded fallback -- native WGSL is
/// not embedded, and "fail loud" trumps papering over Slang failures with
/// stale pre-built WGSL.
class NativeShaderCompiler final : public IShaderCompiler {
   public:
    NativeShaderCompiler(const ShaderLoader& loader, std::filesystem::path cache_dir,
                         std::filesystem::path workspace_root, std::filesystem::path search_path)
        : m_slang(loader, loader.logger(), std::move(cache_dir), std::move(workspace_root),
                  std::move(search_path), /*error_fallback=*/nullptr) {
    }

    std::string compile(const ShaderKey& key) override {
        return m_slang.compile(key);
    }

    std::vector<std::string> poll_dirty() override {
        return m_slang.poll_dirty();
    }

    uint64_t source_revision(std::string_view source_key) const override {
        return m_slang.source_revision(source_key);
    }

    void invalidate(std::string_view source_key) override {
        m_slang.invalidate(source_key);
    }

   private:
    SlangCompiler m_slang;
};
#endif  // __EMSCRIPTEN__

}  // namespace

std::unique_ptr<IShaderCompiler> make_shader_compiler(const ShaderLoader& loader) {
#ifdef __EMSCRIPTEN__
    return std::make_unique<EmbeddedCompiler>(loader);
#else
    namespace fs = std::filesystem;
    std::error_code ec;
    auto exe_dir = fs::current_path(ec);  // fallback; overridden below where possible
#ifdef _WIN32
    {
        wchar_t buf[1024];
        auto n = GetModuleFileNameW(nullptr, buf, 1024);
        if (n > 0) {
            exe_dir = fs::path(buf).parent_path();
        }
    }
#endif
    fs::path cache_dir = exe_dir / "shader_cache";
    fs::path workspace_root = PTS_WORKSPACE_ROOT;
    fs::path search_path = workspace_root / "core" / "shaders";
    return std::make_unique<NativeShaderCompiler>(
        loader, std::move(cache_dir), std::move(workspace_root), std::move(search_path));
#endif
}

}  // namespace pts::rendering
