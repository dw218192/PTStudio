#pragma once

#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/slangCompiler.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace pts::testing {

inline std::optional<std::string_view> stub_getter(std::string_view) {
    return std::string_view{"// stub wgsl"};
}

inline std::filesystem::path unique_cache_dir(const char* tag) {
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "pts_slang_cache_test";
    auto dir = base / (std::string(tag) + "_" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

struct SlangTestCompiler {
    std::filesystem::path cache_dir;
    std::unique_ptr<pts::rendering::SlangCompiler> compiler;

    explicit SlangTestCompiler(pts::rendering::ShaderLoader& loader,
                               std::shared_ptr<spdlog::logger> logger, const char* tag)
        : cache_dir(unique_cache_dir(tag)) {
        std::filesystem::path workspace(PTS_WORKSPACE_ROOT);
        compiler = std::make_unique<pts::rendering::SlangCompiler>(
            loader, std::move(logger), cache_dir, workspace,
            std::vector<std::filesystem::path>{workspace}, nullptr);
    }

    ~SlangTestCompiler() {
        compiler.reset();
        std::error_code ec;
        std::filesystem::remove_all(cache_dir, ec);
    }

    SlangTestCompiler(const SlangTestCompiler&) = delete;
    SlangTestCompiler& operator=(const SlangTestCompiler&) = delete;

    auto get() -> pts::rendering::SlangCompiler* {
        return compiler.get();
    }
};

}  // namespace pts::testing
