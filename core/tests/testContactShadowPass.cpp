#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/contactShadowPass.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/gbufferPass.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderCompiler.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <glm/glm.hpp>

using namespace pts::rendering;

TEST_CASE("profiler init" * doctest::test_suite("setup")) {
    PTS_STARTUP_PROFILER();
}

// Minimal WGSL that satisfies the contact shadow pipeline layout.
// Bindings match GBuffer consumer slots (0-3) + ContactShadow-specific (4-5).
static constexpr auto k_contact_shadow_wgsl = R"(
struct ContactShadowUniforms {
    projection : mat4x4<f32>,
    inv_projection : mat4x4<f32>,
    view : mat4x4<f32>,
    viewport_size : vec2<f32>,
    max_distance : f32,
    thickness : f32,
    normal_offset : f32,
    step_count : i32,
    light_count : u32,
    _pad : u32,
}

struct Light {
    direction_or_pos : vec3<f32>,
    light_type : u32,
    color : vec3<f32>,
    intensity : f32,
    right : vec3<f32>,
    radius : f32,
    up : vec3<f32>,
    angle : f32,
}

@group(0) @binding(0) var depth_tex : texture_depth_2d;
@group(0) @binding(1) var depth_sampler : sampler;
@group(0) @binding(2) var normals_tex : texture_2d<f32>;
@group(0) @binding(3) var linear_sampler : sampler;
@group(0) @binding(4) var<uniform> u : ContactShadowUniforms;
@group(0) @binding(5) var<storage, read> lights : array<Light>;

struct VsOut {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_id : u32) -> VsOut {
    var output : VsOut;
    let uv = vec2<f32>(f32((vertex_id << 1u) & 2u), f32(vertex_id & 2u));
    output.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return output;
}

@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 0.0);
}
)";

// Minimal gbuffer WGSL for creating depth/normals textures.
static constexpr auto k_gbuffer_wgsl = R"(
struct GBufferUniforms {
    mvp : mat4x4<f32>,
    model : mat4x4<f32>,
    view : mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> u : GBufferUniforms;

struct VsIn {
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) color : vec3<f32>,
    @location(3) uv : vec2<f32>,
}

struct VsOut {
    @builtin(position) position : vec4<f32>,
    @location(0) view_normal : vec3<f32>,
}

@vertex
fn vs_main(input : VsIn) -> VsOut {
    var output : VsOut;
    output.position = u.mvp * vec4<f32>(input.position, 1.0);
    output.view_normal = (u.view * u.model * vec4<f32>(input.normal, 0.0)).xyz;
    return output;
}

struct FsOut {
    @location(0) normals : vec4<f32>,
}

@fragment
fn fs_main(@location(0) view_normal : vec3<f32>) -> FsOut {
    let n = normalize(view_normal);
    var output : FsOut;
    output.normals = vec4<f32>(n.xy, 0.0, 1.0);
    return output;
}
)";

namespace {

auto make_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::get("contact_shadow_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("contact_shadow_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}

auto fake_shader_getter(std::string_view key) -> std::optional<std::string_view> {
    if (key == "core/generated/shaders/contact_shadow.wgsl") {
        return k_contact_shadow_wgsl;
    }
    if (key == "core/generated/shaders/gbuffer.wgsl") {
        return k_gbuffer_wgsl;
    }
    return std::nullopt;
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
                           "core/generated/shaders/contact_shadow.wgsl", fake_shader_getter);
    loader.register_shader("core/generated/shaders/gbuffer.wgsl", "core/shaders/gbuffer.slang",
                           "core/generated/shaders/gbuffer.wgsl", fake_shader_getter);

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
                           "core/generated/shaders/contact_shadow.wgsl", fake_shader_getter);
    loader.register_shader("core/generated/shaders/gbuffer.wgsl", "core/shaders/gbuffer.slang",
                           "core/generated/shaders/gbuffer.wgsl", fake_shader_getter);

    GBufferPass gbuf_pass(loader);

    ContactShadowPass cs_pass(loader);

    EmbeddedCompiler compiler(loader);
    FrameGraph fg(device, logger, &compiler);

    OrbitCamera camera;
    RenderWorld world;

    // Add a distant light so the light buffer is non-empty
    {
        auto scope = world.begin_sync();
        auto li = scope.alloc_light_slot();
        auto lw = scope.write_light(li);
        lw->type = LightData::Type::Distant;
        lw->direction = glm::vec3(0, -1, 0);
        lw->color = glm::vec3(1);
        lw->intensity = 1.0f;
    }
    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 800, 600,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    auto gbuf_out = gbuf_pass.add_to_frame_graph(fg, ctx, {});

    auto cs_out =
        cs_pass.add_to_frame_graph(fg, ctx,
                                   {gbuf_out.depth, gbuf_out.normals, world.light_buffer().handle(),
                                    world.light_buffer().size()},
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
                           "core/generated/shaders/contact_shadow.wgsl", fake_shader_getter);
    loader.register_shader("core/generated/shaders/gbuffer.wgsl", "core/shaders/gbuffer.slang",
                           "core/generated/shaders/gbuffer.wgsl", fake_shader_getter);

    GBufferPass gbuf_pass(loader);

    ContactShadowPass cs_pass(loader);

    EmbeddedCompiler compiler(loader);
    FrameGraph fg(device, logger, &compiler);
    cs_pass.m_enabled = false;
    OrbitCamera camera;
    RenderWorld world;
    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 800, 600,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    auto gbuf_out = gbuf_pass.add_to_frame_graph(fg, ctx, {});

    auto cs_out =
        cs_pass.add_to_frame_graph(fg, ctx,
                                   {gbuf_out.depth, gbuf_out.normals, world.light_buffer().handle(),
                                    world.light_buffer().size()},
                                   fg.fallback_pool());

    CHECK(!bool(cs_out.contact_shadow));
}

#endif  // !__EMSCRIPTEN__
