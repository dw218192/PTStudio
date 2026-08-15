#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/gbufferPass.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <glm/glm.hpp>

#include "slangTestSupport.h"

using namespace pts::rendering;

TEST_CASE("profiler init" * doctest::test_suite("setup")) {
    PTS_STARTUP_PROFILER();
}

namespace {

auto make_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::get("gbuffer_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("gbuffer_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}

void register_gbuffer_shader(ShaderLoader& loader) {
    loader.register_shader("core/generated/shaders/gbuffer.wgsl", "core/shaders/gbuffer.slang",
                           "core/generated/shaders/gbuffer.wgsl", pts::testing::stub_getter);
}

}  // namespace

#ifndef __EMSCRIPTEN__

TEST_CASE("GBufferPass exposes Normals + Motion debug targets") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_gbuffer_shader(loader);

    GBufferPass pass(loader);
    pass.ensure_initialized(device);

    auto [targets, count] = pass.debug_targets();
    REQUIRE(count == 2);
    CHECK(targets[0].label == std::string_view("Normals"));
    CHECK(targets[0].resource_name == std::string_view("scene_normals"));
    CHECK(targets[1].label == std::string_view("Motion"));
    CHECK(targets[1].resource_name == std::string_view("scene_motion"));
}

TEST_CASE("GBufferPass produces an RG16Float motion texture alongside depth + normals") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_gbuffer_shader(loader);

    GBufferPass pass(loader);

    pts::testing::SlangTestCompiler slang(loader, logger, "gbuf_outputs");
    FrameGraph fg(device, logger, slang.get());

    OrbitCamera camera;
    RenderWorld world;
    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 256, 192,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    auto out = pass.add_to_frame_graph(fg, ctx, {});
    REQUIRE(bool(out.depth));
    REQUIRE(bool(out.normals));
    REQUIRE(bool(out.motion));

    fg.compile();

    const auto* motion_tex = fg.compiled_texture(out.motion);
    REQUIRE(motion_tex != nullptr);
    CHECK(motion_tex->view != nullptr);
    CHECK(motion_tex->desc.format == WGPUTextureFormat_RG16Float);
    CHECK(motion_tex->desc.width == 256);
    CHECK(motion_tex->desc.height == 192);
}

TEST_CASE("GBufferPass tolerates an empty world (no objects -> no motion entries)") {
    // Regression: an empty world must not trip the per-slot prev-transform
    // bookkeeping when total_slots == 0.
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_gbuffer_shader(loader);

    GBufferPass pass(loader);

    pts::testing::SlangTestCompiler slang(loader, logger, "gbuf_empty");
    FrameGraph fg(device, logger, slang.get());

    OrbitCamera camera;
    RenderWorld world;
    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 64, 64,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    // Two consecutive frames so the second exercises the "prev camera valid"
    // path (prev_view/prev_proj propagated).
    for (int i = 0; i < 2; ++i) {
        fg.begin_frame();
        auto out = pass.add_to_frame_graph(fg, ctx, {});
        REQUIRE(bool(out.motion));
        fg.compile();
    }
}

#endif  // !__EMSCRIPTEN__
