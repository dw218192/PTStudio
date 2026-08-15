#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/contactShadowPass.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/gbufferPass.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderc/shaderLoader.h>
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
    auto logger = spdlog::get("contact_shadow_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("contact_shadow_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}

// Build+register the contact shadow consumer BGL. In production this is
// registered by the owning renderer (forwardPass) from its shader's
// reflection; tests pre-register the canonical shape (sampled_texture<R8>).
void register_cs_consumer_bgl(pts::rendering::FrameGraph& fg, WGPUDevice device) {
    WGPUBindGroupLayoutEntry entries[2]{};
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[0].texture.sampleType = WGPUTextureSampleType_Float;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[0].texture.multisampled = false;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].sampler.type = WGPUSamplerBindingType_Filtering;

    WGPUBindGroupLayoutDescriptor desc{};
    desc.entryCount = 2;
    desc.entries = entries;
    fg.bind_group_layout("contact_shadow/consumer", wgpuDeviceCreateBindGroupLayout(device, &desc));
}

}  // namespace

// --- GPU tests ---

#ifndef __EMSCRIPTEN__

TEST_CASE("ContactShadowPass reports debug target when enabled") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    loader.register_shader("core/generated/shaders/contact_shadow.wgsl",
                           "core/shaders/contact_shadow.slang",
                           "core/generated/shaders/contact_shadow.wgsl", pts::testing::stub_getter);
    loader.register_shader("core/generated/shaders/gbuffer.wgsl", "core/shaders/gbuffer.slang",
                           "core/generated/shaders/gbuffer.wgsl", pts::testing::stub_getter);

    ContactShadowPass pass(loader);
    pass.ensure_initialized(device);

    auto [targets, count] = pass.debug_targets();
    CHECK(count == 1);
    CHECK(targets[0].label == std::string_view("Contact Shadow"));
    CHECK(targets[0].resource_name == std::string_view("contact_shadow"));

    pass.m_enabled = false;
    auto [targets2, count2] = pass.debug_targets();
    CHECK(count2 == 0);
}

TEST_CASE("ContactShadowPass add_to_frame_graph produces valid output") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    loader.register_shader("core/generated/shaders/contact_shadow.wgsl",
                           "core/shaders/contact_shadow.slang",
                           "core/generated/shaders/contact_shadow.wgsl", pts::testing::stub_getter);
    loader.register_shader("core/generated/shaders/gbuffer.wgsl", "core/shaders/gbuffer.slang",
                           "core/generated/shaders/gbuffer.wgsl", pts::testing::stub_getter);

    GBufferPass gbuf_pass(loader);

    ContactShadowPass cs_pass(loader);
    cs_pass.m_blur = false;

    pts::testing::SlangTestCompiler slang(loader, logger, "cs_valid_output");
    FrameGraph fg(device, logger, slang.get());

    OrbitCamera camera;
    RenderWorld world;

    // Add a distant light so the light buffer is non-empty
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

    PassContext ctx{device,       device.queue(), camera,       world, 800, 600,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    register_cs_consumer_bgl(fg, device.handle());
    auto gbuf_out = gbuf_pass.add_to_frame_graph(fg, ctx, {});

    auto cs_out =
        cs_pass.add_to_frame_graph(fg, ctx,
                                   {gbuf_out.depth, gbuf_out.normals, world.light_buffer().handle,
                                    world.light_buffer().size_bytes},
                                   fg.fallback_pool());

    CHECK(bool(cs_out.contact_shadow));

    fg.compile();
    const auto* cs_tex = fg.compiled_texture(cs_out.contact_shadow);
    REQUIRE(cs_tex != nullptr);
    CHECK(cs_tex->view != nullptr);
}

TEST_CASE("ContactShadowPass disabled returns invalid handle") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    loader.register_shader("core/generated/shaders/contact_shadow.wgsl",
                           "core/shaders/contact_shadow.slang",
                           "core/generated/shaders/contact_shadow.wgsl", pts::testing::stub_getter);
    loader.register_shader("core/generated/shaders/gbuffer.wgsl", "core/shaders/gbuffer.slang",
                           "core/generated/shaders/gbuffer.wgsl", pts::testing::stub_getter);

    GBufferPass gbuf_pass(loader);

    ContactShadowPass cs_pass(loader);

    pts::testing::SlangTestCompiler slang(loader, logger, "cs_disabled");
    FrameGraph fg(device, logger, slang.get());
    cs_pass.m_enabled = false;
    OrbitCamera camera;
    RenderWorld world;
    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 800, 600,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    register_cs_consumer_bgl(fg, device.handle());
    auto gbuf_out = gbuf_pass.add_to_frame_graph(fg, ctx, {});

    auto cs_out =
        cs_pass.add_to_frame_graph(fg, ctx,
                                   {gbuf_out.depth, gbuf_out.normals, world.light_buffer().handle,
                                    world.light_buffer().size_bytes},
                                   fg.fallback_pool());

    CHECK(!bool(cs_out.contact_shadow));
}

#endif  // !__EMSCRIPTEN__
