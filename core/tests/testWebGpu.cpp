#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {
auto load_shader_source() -> std::string {
    std::filesystem::path shader_path = PTS_SOURCE_DIR;
    shader_path /= "assets";
    shader_path /= "shaders";
    shader_path /= "test";
    shader_path /= "simple.wgsl";

    std::ifstream file(shader_path, std::ios::binary);
    if (!file) {
        return {};
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

auto create_test_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::get("webgpu_test");
    if (!logger) {
        // Create new logger if it doesn't exist
        logger = spdlog::stdout_color_mt("webgpu_test");
    }
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

    // Buffer creation throws on failure; is_valid() supported for optional buffers
    auto buffer = device.create_buffer(1024, WGPUBufferUsage_Vertex);
    CHECK(buffer.is_valid());

    // ShaderModule factory throws on failure; invariant enforces non-null
    auto shader_source = load_shader_source();
    REQUIRE(!shader_source.empty());
    auto shader = device.create_shader_module_from_source(shader_source);
    CHECK(shader.handle() != nullptr);
}
