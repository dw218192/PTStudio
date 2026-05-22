#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/temporalStorage.h>
#include <core/rendering/temporalVisibilityPass.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <pxr/usd/sdf/path.h>
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
    auto logger = spdlog::get("temporal_visibility_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("temporal_visibility_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}

}  // namespace

// --- TemporalStorageManager unit tests ---

#ifndef __EMSCRIPTEN__

TEST_CASE("TemporalStorageManager::request_persistent dedupes by name") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    pts::testing::SlangTestCompiler slang(loader, logger, "tsm_dedupe");
    FrameGraph fg(device, logger, slang.get());

    fg.begin_frame();

    TemporalStorageManager storage;
    auto h1 = storage.request_persistent(fg, "vis", 256, 128, WGPUTextureFormat_R16Float,
                                         WGPUTextureUsage_TextureBinding);
    auto h2 = storage.request_persistent(fg, "vis", 256, 128, WGPUTextureFormat_R16Float,
                                         WGPUTextureUsage_TextureBinding);
    CHECK(h1 == h2);
    CHECK(bool(h1));
}

TEST_CASE("TemporalStorageManager::request_ping_pong rotates with frame_index parity") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    pts::testing::SlangTestCompiler slang(loader, logger, "tsm_pingpong");
    FrameGraph fg(device, logger, slang.get());

    fg.begin_frame();

    TemporalStorageManager storage;
    auto usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                               WGPUTextureUsage_TextureBinding);

    auto pp_even =
        storage.request_ping_pong(fg, "vis", 64, 64, WGPUTextureFormat_R16Float, usage, 0);
    auto pp_odd =
        storage.request_ping_pong(fg, "vis", 64, 64, WGPUTextureFormat_R16Float, usage, 1);

    // Both handles must be valid and distinct in each frame.
    CHECK(bool(pp_even.read));
    CHECK(bool(pp_even.write));
    CHECK(pp_even.read != pp_even.write);

    // Roles swap on the next frame.
    CHECK(pp_odd.read == pp_even.write);
    CHECK(pp_odd.write == pp_even.read);
}

// --- TemporalVisibilityPass GPU tests ---

TEST_CASE("TemporalVisibilityPass disabled returns invalid texture and fallback consumer") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    loader.register_shader(
        "core/generated/shaders/shadow/temporal.wgsl", "core/shaders/shadow/temporal.slang",
        "core/generated/shaders/shadow/temporal.wgsl", pts::testing::stub_getter);

    TemporalVisibilityPass pass(loader);
    pass.m_enabled = false;

    pts::testing::SlangTestCompiler slang(loader, logger, "tv_disabled");
    FrameGraph fg(device, logger, slang.get());

    OrbitCamera camera;
    RenderWorld world;
    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 320, 240,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();

    TemporalStorageManager storage;
    auto out = pass.add_to_frame_graph(fg, ctx, {{}, {}, {}, UINT32_MAX}, storage);

    CHECK(!bool(out.accumulated_visibility));
}

TEST_CASE("TemporalVisibilityPass produces valid persistent visibility output") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    loader.register_shader(
        "core/generated/shaders/shadow/temporal.wgsl", "core/shaders/shadow/temporal.slang",
        "core/generated/shaders/shadow/temporal.wgsl", pts::testing::stub_getter);
    loader.register_shader(
        "core/generated/shaders/shadow/shadow_map.wgsl", "core/shaders/shadow/shadow_map.slang",
        "core/generated/shaders/shadow/shadow_map.wgsl", pts::testing::stub_getter, {"vs_main"});

    TemporalVisibilityPass pass(loader);
    ShadowMapPass shadow_pass(loader);

    pts::testing::SlangTestCompiler slang(loader, logger, "tv_valid");
    FrameGraph fg(device, logger, slang.get());

    OrbitCamera camera;
    RenderWorld world;

    {
        auto scope = world.begin_sync();
        auto li = scope.alloc_light(pxr::SdfPath("/TestLight0"));
        scope.mutate_light(li, LightField::All, [&](LightData& lw) {
            lw.type = LightData::Type::Distant;
            lw.direction = glm::vec3(0, -1, 0);
            lw.color = glm::vec3(1);
            lw.intensity = 1.0f;
        });
    }
    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 320, 240,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    // Frame 0
    fg.begin_frame();
    auto shadow_out = shadow_pass.add_to_frame_graph(fg, ctx, {});

    TextureDesc depth_desc;
    depth_desc.width = 320;
    depth_desc.height = 240;
    depth_desc.format = WGPUTextureFormat_Depth32Float;
    depth_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                                     WGPUTextureUsage_TextureBinding);
    auto depth_decl = fg.texture("test_depth", depth_desc);

    TemporalStorageManager storage;
    auto out_f0 = pass.add_to_frame_graph(
        fg, ctx, {depth_decl, shadow_out.shadow_array, shadow_out.shadow_info, 0u}, storage);

    CHECK(bool(out_f0.accumulated_visibility));

    fg.compile();
    const auto* tex_f0 = fg.compiled_texture(out_f0.accumulated_visibility);
    REQUIRE(tex_f0 != nullptr);
    CHECK(tex_f0->view != nullptr);
    CHECK(tex_f0->desc.format == WGPUTextureFormat_R16Float);
    CHECK(tex_f0->desc.width == 320);
    CHECK(tex_f0->desc.height == 240);

    // Frame 1: ping-pong should swap the write target. Compare write-target
    // handles across frames -- they must differ (read holds frame 0's data).
    fg.begin_frame();
    auto shadow_out_f1 = shadow_pass.add_to_frame_graph(fg, ctx, {});
    auto depth_decl_f1 = fg.texture("test_depth", depth_desc);
    auto out_f1 = pass.add_to_frame_graph(
        fg, ctx, {depth_decl_f1, shadow_out_f1.shadow_array, shadow_out_f1.shadow_info, 0u},
        storage);

    CHECK(bool(out_f1.accumulated_visibility));
    CHECK(out_f1.accumulated_visibility != out_f0.accumulated_visibility);
}

#endif  // !__EMSCRIPTEN__
