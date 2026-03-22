#include "wireframePass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <core/rendering/wireframeIndices.h>
#include <wireframe_shader_metadata.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace pts;
using namespace pts::editor;
using namespace pts::rendering;

struct WireframeMesh {
    webgpu::Buffer index_buffer;
    uint32_t index_count;
};

struct WireframeUniforms {
    glm::mat4 mvp;
};
static_assert(sizeof(WireframeUniforms) == 64, "WireframeUniforms must match shader std140 layout");
static_assert(WireframePass::k_uniform_align >= sizeof(WireframeUniforms),
              "Alignment must be >= uniform struct size");

static WGPUBindGroup create_bind_group(WGPUDevice device, WGPUBindGroupLayout layout,
                                       WGPUBuffer uniform_buf) {
    WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
    entry.binding = 0;
    entry.buffer = uniform_buf;
    entry.offset = 0;
    entry.size = sizeof(WireframeUniforms);

    WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg_desc.layout = layout;
    bg_desc.entryCount = 1;
    bg_desc.entries = &entry;
    return wgpuDeviceCreateBindGroup(device, &bg_desc);
}

WireframePass::~WireframePass() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group) {
            wgpuBindGroupRelease(ready->bind_group);
        }
        if (ready->bind_group_layout) {
            wgpuBindGroupLayoutRelease(ready->bind_group_layout);
        }
    }
}

auto WireframePass::name() const noexcept -> std::string_view {
    return "wireframe";
}

auto WireframePass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

void WireframePass::do_setup(const webgpu::Device& device) {
    // Capture old state for deferred release (after new state is built)
    WGPUBindGroup old_bind_group = nullptr;
    WGPUBindGroupLayout old_layout = nullptr;
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        old_bind_group = ready->bind_group;
        old_layout = ready->bind_group_layout;
        ready->bind_group = nullptr;
        ready->bind_group_layout = nullptr;
    }
    clear_pass_data();

    auto shader_src = get_shader_loader().load("editor/generated/shaders/wireframe.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    uint32_t initial_capacity = 64;
    auto uniform_buffer = device.create_buffer(
        k_uniform_align * initial_capacity,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    WGPUBindGroupLayoutEntry bgl_entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    bgl_entry.binding = 0;
    bgl_entry.visibility =
        static_cast<WGPUShaderStage>(WGPUShaderStage_Vertex | WGPUShaderStage_Fragment);
    bgl_entry.buffer.type = WGPUBufferBindingType_Uniform;
    bgl_entry.buffer.hasDynamicOffset = true;
    bgl_entry.buffer.minBindingSize = sizeof(WireframeUniforms);

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = 1;
    bgl_desc.entries = &bgl_entry;
    auto bind_group_layout = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);

    auto bind_group =
        create_bind_group(device.handle(), bind_group_layout, uniform_buffer.handle());

    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &bind_group_layout;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

    auto pipeline = webgpu::RenderPipelineBuilder(device)
                        .shader(shader)
                        .color_format(WGPUTextureFormat_RGBA16Float)
                        .depth_format(WGPUTextureFormat_Depth24Plus)
                        .depth_write(true)
                        .depth_compare(WGPUCompareFunction_Less)
                        .cull_mode(WGPUCullMode_None)
                        .topology(WGPUPrimitiveTopology_LineList)
                        .pipeline_layout(pipeline_layout)
                        .vertex_layout<editor_wireframe_shader::VertexLayout>()
                        .build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    m_state = Ready{
        std::move(shader), std::move(pipeline), std::move(uniform_buffer),
        bind_group,        bind_group_layout,   initial_capacity,
    };

    if (old_bind_group) wgpuBindGroupRelease(old_bind_group);
    if (old_layout) wgpuBindGroupLayoutRelease(old_layout);
}

void WireframePass::ensure_capacity(const webgpu::Device& device, uint32_t object_count) {
    auto& ready = std::get<Ready>(m_state);
    if (object_count <= ready.capacity) return;

    uint32_t new_capacity = ready.capacity;
    while (new_capacity < object_count) {
        new_capacity *= 2;
    }

    ready.uniform_buffer = device.create_buffer(
        k_uniform_align * new_capacity,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    if (ready.bind_group) {
        wgpuBindGroupRelease(ready.bind_group);
    }

    ready.bind_group =
        create_bind_group(device.handle(), ready.bind_group_layout, ready.uniform_buffer.handle());
    ready.capacity = new_capacity;
}

void WireframePass::add_to_frame_graph(rendering::FrameGraph& fg,
                                       const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    auto objects = ctx.world.get_objects();
    auto meshes = ctx.world.get_meshes();
    auto object_count = static_cast<uint32_t>(objects.size());
    if (object_count > 0) {
        ensure_capacity(ctx.device, object_count);
    }

    rendering::TextureDesc color_desc;
    color_desc.width = ctx.viewport_width;
    color_desc.height = ctx.viewport_height;
    color_desc.format = WGPUTextureFormat_RGBA16Float;
    color_desc.clear_color = {0.15, 0.15, 0.18, 1.0};

    rendering::TextureDesc depth_desc;
    depth_desc.width = ctx.viewport_width;
    depth_desc.height = ctx.viewport_height;
    depth_desc.format = WGPUTextureFormat_Depth24Plus;

    auto color = fg.find_or_create("scene_color", color_desc);
    auto depth = fg.find_or_create("scene_depth", depth_desc);

    auto queue = ctx.queue;
    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    auto* pipeline_handle = ready.pipeline.handle();
    auto uniform_buf = ready.uniform_buffer.handle();
    auto bind_group = ready.bind_group;
    const auto& world = ctx.world;

    // Lazily build wireframe index buffers via the per-pass mesh cache.
    for (uint32_t i = 0; i < object_count; ++i) {
        if (!objects[i].active()) continue;
        const auto& obj = objects[i];
        get_or_create_pass_data<WireframeMesh>(
            rendering::PassDataKind::Mesh, obj->mesh_index, ctx.world, [&]() {
                const auto& mesh = meshes[obj->mesh_index];
                auto indices = expand_wireframe_indices(mesh->cpu_indices);
                auto buf = ctx.device.create_buffer(
                    indices.size() * sizeof(uint32_t),
                    static_cast<WGPUBufferUsage>(WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst));
                wgpuQueueWriteBuffer(queue, buf.handle(), 0, indices.data(),
                                     indices.size() * sizeof(uint32_t));
                return WireframeMesh{std::move(buf), static_cast<uint32_t>(indices.size())};
            });
    }

    for (uint32_t i = 0; i < object_count; ++i) {
        if (!objects[i].active()) continue;
        const auto& obj = objects[i];
        WireframeUniforms u{};
        u.mvp = proj_mat * view_mat * obj->transform;
        wgpuQueueWriteBuffer(queue, uniform_buf, i * k_uniform_align, &u, sizeof(u));
    }

    fg.add_pass("wireframe")
        .color(color)
        .depth(depth)
        .execute([=, &world](WGPURenderPassEncoder pass) {
            auto objs = world.get_objects();
            auto mshs = world.get_meshes();
            wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
            for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                if (!objs[i].active()) continue;
                uint32_t dyn_offset = i * k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 1, &dyn_offset);
                const auto& mesh = mshs[objs[i]->mesh_index];
                auto& wf = get_or_create_pass_data<WireframeMesh>(
                    rendering::PassDataKind::Mesh, objs[i]->mesh_index, world, nullptr);
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh->vertex_buffer.handle(), 0,
                                                     mesh->vertex_buffer.size());
                wgpuRenderPassEncoderSetIndexBuffer(pass, wf.index_buffer.handle(),
                                                    WGPUIndexFormat_Uint32, 0,
                                                    wf.index_buffer.size());
                wgpuRenderPassEncoderDrawIndexed(pass, wf.index_count, 1, 0, 0, 0);
            }
        });
}
