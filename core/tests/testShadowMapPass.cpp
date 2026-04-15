#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderCompiler.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <pxr/usd/sdf/path.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <glm/glm.hpp>

using namespace pts::rendering;

// Tracy profiler must be initialized for instrumented code paths (PTS_ZONE_SCOPED).
TEST_CASE("profiler init" * doctest::test_suite("setup")) {
    PTS_STARTUP_PROFILER();
}

// Minimal shadow-compatible WGSL: position-only vertex shader, no fragment.
static constexpr auto k_shadow_wgsl = R"(
struct ShadowPerObject {
    light_mvp : mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> u : ShadowPerObject;

struct VsIn {
    @location(0) position : vec3<f32>,
}

struct VsOut {
    @builtin(position) position : vec4<f32>,
}

@vertex
fn vs_main(input : VsIn) -> VsOut {
    var output : VsOut;
    output.position = u.light_mvp * vec4<f32>(input.position, 1.0);
    return output;
}
)";

namespace {

auto make_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::get("shadow_map_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("shadow_map_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}

auto fake_shader_getter(std::string_view key) -> std::optional<std::string_view> {
    if (key == "core/generated/shaders/shadow.wgsl") {
        return k_shadow_wgsl;
    }
    return std::nullopt;
}

// Build+register the shadow consumer BGL the pass expects. In production
// this is registered by the owning renderer (forwardPass) from its shader's
// reflection; tests don't depend on forward, so we construct it explicitly
// to match the canonical shape: storage(ShadowInfo) + texture2DArray(depth)
// + sampler(non-filtering).
void register_shadow_consumer_bgl(pts::rendering::FrameGraph& fg, WGPUDevice device) {
    WGPUBindGroupLayoutEntry entries[3]{};
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].texture.sampleType = WGPUTextureSampleType_Depth;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_2DArray;
    entries[1].texture.multisampled = false;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Fragment;
    entries[2].sampler.type = WGPUSamplerBindingType_NonFiltering;

    WGPUBindGroupLayoutDescriptor desc{};
    desc.entryCount = 3;
    desc.entries = entries;
    fg.bind_group_layout("shadow_map/consumer", wgpuDeviceCreateBindGroupLayout(device, &desc));
}

}  // namespace

// --- GPU tests ---

#ifndef __EMSCRIPTEN__

TEST_CASE("ShadowMapPass add_to_frame_graph with no lights returns valid handles") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    loader.register_shader("core/generated/shaders/shadow.wgsl", "core/shaders/shadow.slang",
                           "core/generated/shaders/shadow.wgsl", fake_shader_getter, {"vs_main"});

    ShadowMapPass pass(loader);

    EmbeddedCompiler compiler(loader);
    FrameGraph fg(device, logger, &compiler);

    OrbitCamera camera;
    RenderWorld world;

    PassContext ctx{device,       device.queue(), camera,       world, 800, 600,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    register_shadow_consumer_bgl(fg, device.handle());
    auto out = pass.add_to_frame_graph(fg, ctx, {});

    CHECK(bool(out.shadow_array));
    CHECK(bool(out.shadow_info));
}

TEST_CASE("ShadowMapPass add_to_frame_graph with distant light produces valid outputs") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    loader.register_shader("core/generated/shaders/shadow.wgsl", "core/shaders/shadow.slang",
                           "core/generated/shaders/shadow.wgsl", fake_shader_getter, {"vs_main"});

    ShadowMapPass pass(loader);

    EmbeddedCompiler compiler(loader);
    FrameGraph fg(device, logger, &compiler);

    OrbitCamera camera;
    RenderWorld world;

    // Add a distant light
    {
        auto scope = world.begin_sync();
        auto li = scope.alloc_light(pxr::SdfPath("/TestLight0"));
        scope.mutate_light(li, [&](LightData& lw) {
            lw.type = LightData::Type::Distant;
            lw.direction = glm::vec3(0, -1, 0);
            lw.color = glm::vec3(1);
            lw.intensity = 1.0f;
        });
    }

    // Add a mesh with some geometry
    uint32_t mesh_idx;
    {
        auto scope = world.begin_sync();
        mesh_idx = scope.alloc_mesh(pxr::SdfPath("/TestMesh0"));
        scope.mutate_mesh(mesh_idx, [&](MeshData& mw) {
            mw.cpu_vertices = {
                {{-1, -1, -1}, {0, 1, 0}, {1, 1, 1}, {0, 0}},
                {{1, -1, -1}, {0, 1, 0}, {1, 1, 1}, {1, 0}},
                {{0, 1, 0}, {0, 1, 0}, {1, 1, 1}, {0.5f, 1}},
            };
            mw.cpu_indices = {0, 1, 2};
            mw.index_count = 3;
        });
    }
    world.upload_all_meshes(device);

    // Add an object referencing the mesh
    {
        auto scope = world.begin_sync();
        auto oi = scope.alloc_object(pxr::SdfPath("/TestObj0"));
        scope.mutate_object(oi, [&](ObjectData& ow) {
            ow.mesh_index = mesh_idx;
            ow.transform = glm::mat4(1.0f);
        });
    }

    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 800, 600,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    register_shadow_consumer_bgl(fg, device.handle());
    auto out = pass.add_to_frame_graph(fg, ctx, {});

    CHECK(bool(out.shadow_array));
    CHECK(bool(out.shadow_info));

    // Compile and execute to verify resources are properly allocated
    fg.compile();
    const auto* shadow_tex = fg.compiled_texture(out.shadow_array);
    const auto* shadow_buf = fg.compiled_buffer(out.shadow_info);
    REQUIRE(shadow_tex != nullptr);
    CHECK(shadow_tex->view != nullptr);
    REQUIRE(shadow_buf != nullptr);
    CHECK(shadow_buf->buffer != nullptr);
}

TEST_CASE("ShadowMapPass caps shadow count at k_max_shadow_maps") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    loader.register_shader("core/generated/shaders/shadow.wgsl", "core/shaders/shadow.slang",
                           "core/generated/shaders/shadow.wgsl", fake_shader_getter, {"vs_main"});

    ShadowMapPass pass(loader);

    EmbeddedCompiler compiler(loader);
    FrameGraph fg(device, logger, &compiler);

    OrbitCamera camera;
    RenderWorld world;

    // Add more distant lights than k_max_shadow_maps
    {
        auto scope = world.begin_sync();
        for (uint32_t i = 0; i < k_max_shadow_maps + 2; ++i) {
            auto li = scope.alloc_light(pxr::SdfPath("/TestLight" + std::to_string(i)));
            scope.mutate_light(li, [&](LightData& lw) {
                lw.type = LightData::Type::Distant;
                lw.direction = glm::vec3(0, -1, 0);
            });
        }
    }

    // Add geometry
    uint32_t mesh_idx;
    {
        auto scope = world.begin_sync();
        mesh_idx = scope.alloc_mesh(pxr::SdfPath("/TestMesh0"));
        scope.mutate_mesh(mesh_idx, [&](MeshData& mw) {
            mw.cpu_vertices = {
                {{-1, -1, -1}, {0, 1, 0}, {1, 1, 1}, {0, 0}},
                {{1, -1, -1}, {0, 1, 0}, {1, 1, 1}, {1, 0}},
                {{0, 1, 0}, {0, 1, 0}, {1, 1, 1}, {0.5f, 1}},
            };
            mw.cpu_indices = {0, 1, 2};
            mw.index_count = 3;
        });
    }
    world.upload_all_meshes(device);

    {
        auto scope = world.begin_sync();
        auto oi = scope.alloc_object(pxr::SdfPath("/TestObj0"));
        scope.mutate_object(oi, [&](ObjectData& ow) {
            ow.mesh_index = mesh_idx;
            ow.transform = glm::mat4(1.0f);
        });
    }

    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 800, 600,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    register_shadow_consumer_bgl(fg, device.handle());
    auto out = pass.add_to_frame_graph(fg, ctx, {});

    CHECK(bool(out.shadow_array));
    CHECK(bool(out.shadow_info));

    // Compile to verify the shadow texture array has the right layer count
    fg.compile();
    const auto* shadow_tex = fg.compiled_texture(out.shadow_array);
    REQUIRE(shadow_tex != nullptr);
    CHECK(shadow_tex->layer_views.size() == k_max_shadow_maps);
}

TEST_CASE("ShadowMapPass skips non-distant lights") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    ShaderLoader loader(logger);
    loader.register_shader("core/generated/shaders/shadow.wgsl", "core/shaders/shadow.slang",
                           "core/generated/shaders/shadow.wgsl", fake_shader_getter, {"vs_main"});

    ShadowMapPass pass(loader);

    EmbeddedCompiler compiler(loader);
    FrameGraph fg(device, logger, &compiler);

    OrbitCamera camera;
    RenderWorld world;

    // Add only non-distant lights (sphere, rect)
    {
        auto scope = world.begin_sync();
        auto l1 = scope.alloc_light(pxr::SdfPath("/TestLight0"));
        scope.mutate_light(l1, [&](LightData& lw1) { lw1.type = LightData::Type::Sphere; });

        auto l2 = scope.alloc_light(pxr::SdfPath("/TestLight1"));
        scope.mutate_light(l2, [&](LightData& lw2) { lw2.type = LightData::Type::Rect; });
    }

    world.prepare_gpu_buffers(device, device.queue());

    PassContext ctx{device,       device.queue(), camera,       world, 800, 600,
                    glm::mat4(1), glm::mat4(1),   glm::vec3(0), 0.0f,  0};

    fg.begin_frame();
    register_shadow_consumer_bgl(fg, device.handle());
    auto out = pass.add_to_frame_graph(fg, ctx, {});

    CHECK(bool(out.shadow_array));
    CHECK(bool(out.shadow_info));

    // Non-distant lights produce a 1-layer fallback array texture
    fg.compile();
    const auto* shadow_tex = fg.compiled_texture(out.shadow_array);
    REQUIRE(shadow_tex != nullptr);
    CHECK(shadow_tex->layer_views.size() == 1);
}

#endif  // !__EMSCRIPTEN__
