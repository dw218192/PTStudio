#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/shadowVisibilityPass.h>
#include <core/rendering/temporalResolvePass.h>
#include <core/rendering/temporalStorage.h>
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
    auto logger = spdlog::get("shadow_visibility_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("shadow_visibility_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}

#ifndef __EMSCRIPTEN__

// Register the gen + resolve + shadow-map shaders so the FrameGraph can
// resolve them through the test Slang compiler.
void register_pipeline_shaders(ShaderLoader& loader) {
    loader.register_shader("core/generated/shaders/shadow/shadow_visibility.wgsl",
                           "core/shaders/shadow/shadow_visibility.slang",
                           "core/generated/shaders/shadow/shadow_visibility.wgsl",
                           pts::testing::stub_getter);
    loader.register_shader("core/generated/shaders/shadow/temporal_resolve.wgsl",
                           "core/shaders/shadow/temporal_resolve.slang",
                           "core/generated/shaders/shadow/temporal_resolve.wgsl",
                           pts::testing::stub_getter);
    loader.register_shader(
        "core/generated/shaders/shadow/shadow_map.wgsl", "core/shaders/shadow/shadow_map.slang",
        "core/generated/shaders/shadow/shadow_map.wgsl", pts::testing::stub_getter, {"vs_main"});
}

// A world with one shadow-casting distant light, so ShadowMapPass emits a
// valid shadow array + ShadowInfo buffer.
void populate_shadow_world(pts::webgpu::Device& device, RenderWorld& world) {
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
}

#endif  // !__EMSCRIPTEN__

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

// --- ShadowVisibilityPass (gen) GPU tests ---

TEST_CASE("ShadowVisibilityPass disabled returns invalid texture") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_pipeline_shaders(loader);

    ShadowVisibilityPass pass(loader);
    pass.m_enabled = false;

    pts::testing::SlangTestCompiler slang(loader, logger, "sv_disabled");
    FrameGraph fg(device, logger, slang.get());

    OrbitCamera camera;
    RenderWorld world;
    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 320, 240,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    auto out = pass.add_to_frame_graph(fg, ctx, {{}, {}, {}, 0u});
    CHECK(!bool(out.raw_visibility));
}

TEST_CASE("ShadowVisibilityPass with no shadow light returns invalid texture") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_pipeline_shaders(loader);

    ShadowVisibilityPass pass(loader);

    pts::testing::SlangTestCompiler slang(loader, logger, "sv_nolight");
    FrameGraph fg(device, logger, slang.get());

    OrbitCamera camera;
    RenderWorld world;
    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 320, 240,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    // UINT32_MAX index -> "no shadow-casting light" -> invalid output even
    // when enabled and given valid depth/shadow inputs.
    TextureDesc depth_desc;
    depth_desc.width = 320;
    depth_desc.height = 240;
    depth_desc.format = WGPUTextureFormat_Depth32Float;
    depth_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                                     WGPUTextureUsage_TextureBinding);
    auto depth_decl = fg.texture("test_depth", depth_desc);

    auto out = pass.add_to_frame_graph(fg, ctx, {depth_decl, {}, {}, UINT32_MAX});
    CHECK(!bool(out.raw_visibility));
}

TEST_CASE("ShadowVisibilityPass produces a raw R16Float visibility texture") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_pipeline_shaders(loader);

    ShadowVisibilityPass pass(loader);
    ShadowMapPass shadow_pass(loader);

    pts::testing::SlangTestCompiler slang(loader, logger, "sv_valid");
    FrameGraph fg(device, logger, slang.get());

    OrbitCamera camera;
    RenderWorld world;
    populate_shadow_world(device, world);

    PassContext ctx{device,       device.queue(), camera,       world, 320, 240,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    auto shadow_out = shadow_pass.add_to_frame_graph(fg, ctx, {});

    TextureDesc depth_desc;
    depth_desc.width = 320;
    depth_desc.height = 240;
    depth_desc.format = WGPUTextureFormat_Depth32Float;
    depth_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                                     WGPUTextureUsage_TextureBinding);
    auto depth_decl = fg.texture("test_depth", depth_desc);

    auto out = pass.add_to_frame_graph(
        fg, ctx, {depth_decl, shadow_out.shadow_array, shadow_out.shadow_info, 0u});
    REQUIRE(bool(out.raw_visibility));

    fg.compile();
    const auto* tex = fg.compiled_texture(out.raw_visibility);
    REQUIRE(tex != nullptr);
    CHECK(tex->view != nullptr);
    CHECK(tex->desc.format == WGPUTextureFormat_R16Float);
    CHECK(tex->desc.width == 320);
    CHECK(tex->desc.height == 240);
}

// --- TemporalResolvePass GPU tests ---

TEST_CASE("TemporalResolvePass with no raw input returns invalid texture") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_pipeline_shaders(loader);

    TemporalResolvePass pass(loader);

    pts::testing::SlangTestCompiler slang(loader, logger, "tr_noinput");
    FrameGraph fg(device, logger, slang.get());

    OrbitCamera camera;
    RenderWorld world;
    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 320, 240,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    TemporalStorageManager storage;
    auto out = pass.add_to_frame_graph(fg, ctx, {{}}, storage);
    CHECK(!bool(out.resolved_visibility));
}

TEST_CASE("TemporalResolvePass disabled passes the raw texture through unchanged") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_pipeline_shaders(loader);

    TemporalResolvePass pass(loader);
    pass.m_enabled = false;

    pts::testing::SlangTestCompiler slang(loader, logger, "tr_disabled");
    FrameGraph fg(device, logger, slang.get());

    OrbitCamera camera;
    RenderWorld world;
    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 320, 240,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();

    TextureDesc raw_desc;
    raw_desc.width = 320;
    raw_desc.height = 240;
    raw_desc.format = WGPUTextureFormat_R16Float;
    raw_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                                   WGPUTextureUsage_TextureBinding);
    auto raw_decl = fg.texture("test_raw_vis", raw_desc);

    TemporalStorageManager storage;
    auto out = pass.add_to_frame_graph(fg, ctx, {raw_decl}, storage);
    CHECK(out.resolved_visibility == raw_decl);
}

TEST_CASE("TemporalResolvePass produces a persistent ping-pong resolved texture") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_pipeline_shaders(loader);

    ShadowVisibilityPass gen_pass(loader);
    TemporalResolvePass resolve_pass(loader);
    ShadowMapPass shadow_pass(loader);

    pts::testing::SlangTestCompiler slang(loader, logger, "tr_valid");
    FrameGraph fg(device, logger, slang.get());

    OrbitCamera camera;
    RenderWorld world;
    populate_shadow_world(device, world);

    PassContext ctx{device,       device.queue(), camera,       world, 320, 240,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    TextureDesc depth_desc;
    depth_desc.width = 320;
    depth_desc.height = 240;
    depth_desc.format = WGPUTextureFormat_Depth32Float;
    depth_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                                     WGPUTextureUsage_TextureBinding);

    TemporalStorageManager storage;

    // Frame 0: gen -> resolve.
    fg.begin_frame();
    auto shadow_out = shadow_pass.add_to_frame_graph(fg, ctx, {});
    auto depth_decl = fg.texture("test_depth", depth_desc);
    auto raw_f0 = gen_pass.add_to_frame_graph(
        fg, ctx, {depth_decl, shadow_out.shadow_array, shadow_out.shadow_info, 0u});
    REQUIRE(bool(raw_f0.raw_visibility));
    auto resolved_f0 = resolve_pass.add_to_frame_graph(fg, ctx, {raw_f0.raw_visibility}, storage);
    REQUIRE(bool(resolved_f0.resolved_visibility));

    fg.compile();
    const auto* tex_f0 = fg.compiled_texture(resolved_f0.resolved_visibility);
    REQUIRE(tex_f0 != nullptr);
    CHECK(tex_f0->view != nullptr);
    CHECK(tex_f0->desc.format == WGPUTextureFormat_R16Float);
    CHECK(tex_f0->desc.width == 320);
    CHECK(tex_f0->desc.height == 240);

    // Frame 1: the ping-pong write target must differ from frame 0's (frame
    // 0's resolved output is now the history that frame 1 reads).
    fg.begin_frame();
    auto shadow_out_f1 = shadow_pass.add_to_frame_graph(fg, ctx, {});
    auto depth_decl_f1 = fg.texture("test_depth", depth_desc);
    auto raw_f1 = gen_pass.add_to_frame_graph(
        fg, ctx, {depth_decl_f1, shadow_out_f1.shadow_array, shadow_out_f1.shadow_info, 0u});
    REQUIRE(bool(raw_f1.raw_visibility));
    auto resolved_f1 = resolve_pass.add_to_frame_graph(fg, ctx, {raw_f1.raw_visibility}, storage);
    REQUIRE(bool(resolved_f1.resolved_visibility));
    CHECK(resolved_f1.resolved_visibility != resolved_f0.resolved_visibility);
}

#endif  // !__EMSCRIPTEN__
