#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>

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
}  // namespace

TEST_CASE("WebGPU - Device init and basic resources") {
    auto device = pts::webgpu::Device::create();
    REQUIRE(device.is_valid());

    auto buffer = device.create_buffer(1024, WGPUBufferUsage_Vertex);
    CHECK(buffer.is_valid());

    auto shader = device.create_shader_module(get_shader_path());
    CHECK(shader.is_valid());
}
