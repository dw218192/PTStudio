#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/rendering/shaderCompiler.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <generated/embedded_test_resources.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace {
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

    // ShaderModule factory throws on failure; invariant enforces non-null.
    // Source WGSL is obtained through the IShaderCompiler interface so the
    // test uses the same code path as production render passes.
    pts::rendering::ShaderLoader loader(logger);
    loader.register_shader("shaders/test/simple.wgsl", "assets/shaders/test/simple.slang",
                           "shaders/test/simple.wgsl", test_resources::get_resource,
                           {"vertex_main"});
    pts::rendering::EmbeddedCompiler compiler(loader);
    auto shader_source = compiler.compile(pts::rendering::ShaderKey{"shaders/test/simple.wgsl"});
    auto shader = device.create_shader_module_from_source(shader_source);
    CHECK(shader.handle() != nullptr);
}
