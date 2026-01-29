#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace {
auto get_shader_path() -> std::string {
    std::filesystem::path shader_path = PTS_SOURCE_DIR;
    shader_path /= "assets";
    shader_path /= "shaders";
    shader_path /= "test";
    shader_path /= "simple.wgsl";
    return shader_path.string();
}

auto create_test_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::stdout_color_mt("webgpu_test");
    logger->set_level(spdlog::level::debug);
    return logger;
}
}  // namespace

TEST_CASE("WebGPU - Device init and basic resources") {
    auto logger = create_test_logger();
    
    // Device::create() throws on failure, so if it returns, device is valid
    auto device = pts::webgpu::Device::create(logger);
    CHECK(device.handle() != nullptr);
    CHECK(device.queue() != nullptr);

    // Buffer and shader creation also throw on failure
    auto buffer = device.create_buffer(1024, WGPUBufferUsage_Vertex);
    CHECK(buffer.is_valid());

    auto shader = device.create_shader_module(get_shader_path());
    CHECK(shader.is_valid());
}
