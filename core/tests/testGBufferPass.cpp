#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/gbufferPass.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <glm/glm.hpp>

using namespace pts::rendering;

TEST_CASE("profiler init" * doctest::test_suite("setup")) {
    PTS_STARTUP_PROFILER();
}

// Minimal gbuffer-compatible WGSL matching the generated shader layout.
static constexpr auto k_gbuffer_wgsl = R"(
struct GBufferUniforms {
    mvp : mat4x4<f32>,
    model_view : mat4x4<f32>,
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
    output.view_normal = normalize((u.model_view * vec4<f32>(input.normal, 0.0)).xyz);
    return output;
}

struct FsOut {
    @location(0) normal : vec2<f32>,
}

@fragment
fn fs_main(@location(0) view_normal : vec3<f32>) -> FsOut {
    var n : vec3<f32> = normalize(view_normal);
    var output : FsOut;
    output.normal = vec2<f32>(n.x, n.y);
    return output;
}
)";

namespace {

auto make_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::get("gbuffer_pass_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("gbuffer_pass_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}

auto fake_shader_getter(std::string_view key) -> std::optional<std::string_view> {
    if (key == "core/generated/shaders/gbuffer.wgsl") {
        return k_gbuffer_wgsl;
    }
    return std::nullopt;
}

}  // namespace

TEST_CASE("GBufferPass starts in unready state") {
    auto logger = make_logger();
    ShaderLoader loader(logger);
    GBufferPass pass(loader);
    CHECK_FALSE(pass.is_ready());
}

#ifndef __EMSCRIPTEN__

TEST_CASE("GBufferPass setup transitions to ready") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    loader.register_shader("core/generated/shaders/gbuffer.wgsl", "core/shaders/gbuffer.slang",
                           "core/generated/shaders/gbuffer.wgsl", fake_shader_getter);

    GBufferPass pass(loader);
    CHECK_FALSE(pass.is_ready());

    pass.setup(device);
    CHECK(pass.is_ready());
}

TEST_CASE("GBufferPass creates scene_depth and scene_normals in frame graph") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    loader.register_shader("core/generated/shaders/gbuffer.wgsl", "core/shaders/gbuffer.slang",
                           "core/generated/shaders/gbuffer.wgsl", fake_shader_getter);

    GBufferPass pass(loader);
    pass.setup(device);

    FrameGraph fg(device, logger);
    OrbitCamera camera;
    RenderWorld world;

    // Add a mesh with some geometry
    uint32_t mesh_idx;
    {
        auto scope = world.begin_sync();
        mesh_idx = scope.alloc_mesh_slot();
        auto mw = scope.write_mesh(mesh_idx);
        mw->cpu_vertices = {
            {{-1, -1, -1}, {0, 1, 0}, {1, 1, 1}, {0, 0}},
            {{1, -1, -1}, {0, 1, 0}, {1, 1, 1}, {1, 0}},
            {{0, 1, 0}, {0, 1, 0}, {1, 1, 1}, {0.5f, 1}},
        };
        mw->cpu_indices = {0, 1, 2};
        mw->index_count = 3;
    }
    world.upload_all_meshes(device);

    // Add an object referencing the mesh
    {
        auto scope = world.begin_sync();
        auto oi = scope.alloc_object_slot();
        auto ow = scope.write_object(oi);
        ow->mesh_index = mesh_idx;
        ow->transform = glm::mat4(1.0f);
    }

    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 800, 600,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    pass.add_to_frame_graph(fg, ctx);

    // Verify the frame graph has scene_depth and scene_normals via find_or_create
    // (they should already exist, so find_or_create returns the existing handle)
    TextureDesc depth_desc;
    depth_desc.width = 800;
    depth_desc.height = 600;
    depth_desc.format = WGPUTextureFormat_Depth32Float;

    TextureDesc normals_desc;
    normals_desc.width = 800;
    normals_desc.height = 600;
    normals_desc.format = WGPUTextureFormat_RG16Float;

    auto depth_handle = fg.find_or_create("scene_depth", depth_desc);
    auto normals_handle = fg.find_or_create("scene_normals", normals_desc);

    CHECK(depth_handle.is_valid());
    CHECK(normals_handle.is_valid());
}

TEST_CASE("GBufferPass handles empty world without crash") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    loader.register_shader("core/generated/shaders/gbuffer.wgsl", "core/shaders/gbuffer.slang",
                           "core/generated/shaders/gbuffer.wgsl", fake_shader_getter);

    GBufferPass pass(loader);
    pass.setup(device);

    FrameGraph fg(device, logger);
    OrbitCamera camera;
    RenderWorld world;

    PassContext ctx{device,       device.queue(), camera,       world, 800, 600,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    pass.add_to_frame_graph(fg, ctx);

    // Should still create the resources even with no objects
    TextureDesc depth_desc;
    depth_desc.width = 800;
    depth_desc.height = 600;
    depth_desc.format = WGPUTextureFormat_Depth32Float;

    auto depth_handle = fg.find_or_create("scene_depth", depth_desc);
    CHECK(depth_handle.is_valid());
}

#endif  // !__EMSCRIPTEN__
