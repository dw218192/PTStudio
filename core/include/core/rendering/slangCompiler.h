#pragma once

// libslang-backed compiler is native-only. Guard the whole header so WASM
// translation units cannot accidentally take a dependency on it.
#ifndef __EMSCRIPTEN__

#include <core/rendering/shaderCompiler.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace spdlog {
class logger;
}

namespace pts::rendering {

class ShaderLoader;

/// libslang-backed IShaderCompiler with on-disk cache + mtime watcher.
///
/// Invoked by FrameGraph to produce WGSL for registered `.slang` sources.
/// Results are cached at `<cache_dir>/<hash(inputs)>.wgsl`, keyed by:
///   source_bytes + dep_file_hashes + sorted_defines + slang_version
///   + target_profile + cache_format_version.
///
/// Threading:
///   `compile()` uses a per-source-key mutex so parallel calls for the same
///   key do not race on the disk cache. Concurrent calls for different keys
///   proceed in parallel.
///
/// Error handling:
///   On compile failure, logs a warning and delegates to `error_fallback`
///   (typically EmbeddedCompiler). Never throws; exceptions at the ABI
///   boundary are a spec violation.
class SlangCompiler final : public IShaderCompiler {
   public:
    SlangCompiler(const ShaderLoader& loader, std::shared_ptr<spdlog::logger> logger,
                  std::filesystem::path cache_dir, std::filesystem::path workspace_root,
                  std::filesystem::path search_path, IShaderCompiler* error_fallback);
    ~SlangCompiler() override;

    SlangCompiler(const SlangCompiler&) = delete;
    SlangCompiler& operator=(const SlangCompiler&) = delete;

    std::string compile(const ShaderKey& key) override;

    std::vector<std::string> poll_dirty() override;

    [[nodiscard]] uint64_t source_revision(std::string_view source_key) const override;

    void invalidate(std::string_view source_key) override;

   private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace pts::rendering

#endif  // !__EMSCRIPTEN__
