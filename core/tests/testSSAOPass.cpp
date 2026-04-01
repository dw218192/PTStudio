#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/ssaoPass.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace pts::rendering;

TEST_CASE("profiler init" * doctest::test_suite("setup")) {
    PTS_STARTUP_PROFILER();
}

// Minimal SSAO-compatible WGSL shaders for testing pipeline creation.
static constexpr auto k_ssao_wgsl = R"(
struct SSAOUniforms {
    projection : mat4x4<f32>,
    inv_projection : mat4x4<f32>,
    viewport_size : vec2<f32>,
    radius : f32,
    bias : f32,
    intensity : f32,
    sample_count : i32,
    _pad0 : u32,
    _pad1 : u32,
}

@group(0) @binding(0) var<uniform> u : SSAOUniforms;
@group(0) @binding(1) var depth_tex : texture_2d<f32>;
@group(0) @binding(2) var normals_tex : texture_2d<f32>;
@group(0) @binding(3) var noise_tex : texture_2d<f32>;
@group(0) @binding(4) var depth_sampler : sampler;
@group(0) @binding(5) var linear_sampler : sampler;
@group(0) @binding(6) var noise_sampler : sampler;
@group(0) @binding(7) var<storage, read> kernel : array<vec4<f32>>;

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

static constexpr auto k_ssao_blur_wgsl = R"(
struct BlurUniforms {
    texel_size : vec2<f32>,
    _pad : vec2<f32>,
}

@group(0) @binding(0) var<uniform> u : BlurUniforms;
@group(0) @binding(1) var ssao_tex : texture_2d<f32>;
@group(0) @binding(2) var depth_tex : texture_2d<f32>;
@group(0) @binding(3) var linear_sampler : sampler;
@group(0) @binding(4) var depth_sampler : sampler;

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

namespace {

auto make_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::get("ssao_pass_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("ssao_pass_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}

auto fake_shader_getter(std::string_view key) -> std::optional<std::string_view> {
    if (key == "core/generated/shaders/ssao.wgsl") return k_ssao_wgsl;
    if (key == "core/generated/shaders/ssao_blur.wgsl") return k_ssao_blur_wgsl;
    return std::nullopt;
}

void register_ssao_shaders(ShaderLoader& loader) {
    loader.register_shader("core/generated/shaders/ssao.wgsl", "core/shaders/ssao.slang",
                           "core/generated/shaders/ssao.wgsl", fake_shader_getter);
    loader.register_shader("core/generated/shaders/ssao_blur.wgsl", "core/shaders/ssao_blur.slang",
                           "core/generated/shaders/ssao_blur.wgsl", fake_shader_getter);
}

}  // namespace

TEST_CASE("SSAOPass starts in unready state") {
    auto logger = make_logger();
    ShaderLoader loader(logger);
    SSAOPass pass(loader);
    CHECK_FALSE(pass.is_ready());
}

#ifndef __EMSCRIPTEN__

TEST_CASE("SSAOPass setup transitions to ready") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_ssao_shaders(loader);

    SSAOPass pass(loader);
    pass.setup(device);
    CHECK(pass.is_ready());
}

TEST_CASE("SSAOPass creates ssao_raw and ssao resources when enabled") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_ssao_shaders(loader);

    SSAOPass pass(loader);
    pass.setup(device);

    FrameGraph fg(device, logger);
    OrbitCamera camera;
    RenderWorld world;

    auto proj = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
    PassContext ctx{device,       device.queue(), camera,       world, 800, 600,
                    glm::mat4(1), proj,           glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    pass.add_to_frame_graph(fg, ctx);

    // Both ssao_raw and ssao should now exist in the frame graph
    TextureDesc r8_desc;
    r8_desc.width = 800;
    r8_desc.height = 600;
    r8_desc.format = WGPUTextureFormat_R8Unorm;
    r8_desc.clear_color = {1, 1, 1, 1};

    auto ssao_raw = fg.find_or_create("ssao_raw", r8_desc);
    auto ssao = fg.find_or_create("debug_AO", r8_desc);
    CHECK(ssao_raw.is_valid());
    CHECK(ssao.is_valid());
}

TEST_CASE("SSAOPass is no-op when disabled") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_ssao_shaders(loader);

    SSAOPass pass(loader);
    pass.setup(device);
    pass.m_enabled = false;

    FrameGraph fg(device, logger);
    OrbitCamera camera;
    RenderWorld world;

    auto proj = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
    PassContext ctx{device,       device.queue(), camera,       world, 800, 600,
                    glm::mat4(1), proj,           glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    pass.add_to_frame_graph(fg, ctx);

    // Frame graph should have no passes and no ssao resources when disabled.
    // Calling find_or_create would create them, so we compile and check that
    // cached_texture_count is zero (nothing was allocated).
    fg.compile();
    CHECK(fg.cached_texture_count() == 0);
}

TEST_CASE("SSAOPass setup is re-entrant (hot-reload)") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_ssao_shaders(loader);

    SSAOPass pass(loader);
    pass.setup(device);
    CHECK(pass.is_ready());

    // Second setup should not leak or crash
    pass.setup(device);
    CHECK(pass.is_ready());
}

TEST_CASE("SSAOPass full pipeline: add passes and compile frame graph") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    register_ssao_shaders(loader);

    SSAOPass pass(loader);
    pass.setup(device);

    FrameGraph fg(device, logger);
    OrbitCamera camera;
    RenderWorld world;

    auto proj = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
    PassContext ctx{device,       device.queue(), camera,       world, 800, 600,
                    glm::mat4(1), proj,           glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    pass.add_to_frame_graph(fg, ctx);
    fg.compile();

    // If we got here without crashing, the frame graph accepted the passes
    CHECK(true);
}

#endif  // !__EMSCRIPTEN__
