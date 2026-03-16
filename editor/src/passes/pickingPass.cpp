#include "pickingPass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <picking_shader_metadata.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "editorResources.h"

using namespace pts;
using namespace pts::editor;

struct PickingUniforms {
    glm::mat4 mvp;
    uint32_t object_id;
    uint32_t _pad[3];
};
static_assert(sizeof(PickingUniforms) == 80, "PickingUniforms must match shader std140 layout");
static_assert(PickingPass::k_uniform_align >= sizeof(PickingUniforms),
              "Alignment must be >= uniform struct size");

static WGPUBindGroup create_bind_group(WGPUDevice device, WGPUBindGroupLayout layout,
                                       WGPUBuffer uniform_buf) {
    WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
    entry.binding = 0;
    entry.buffer = uniform_buf;
    entry.offset = 0;
    entry.size = sizeof(PickingUniforms);

    WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg_desc.layout = layout;
    bg_desc.entryCount = 1;
    bg_desc.entries = &entry;
    return wgpuDeviceCreateBindGroup(device, &bg_desc);
}

PickingPass::~PickingPass() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group) {
            wgpuBindGroupRelease(ready->bind_group);
        }
        if (ready->bind_group_layout) {
            wgpuBindGroupLayoutRelease(ready->bind_group_layout);
        }
    }
}

auto PickingPass::name() const noexcept -> std::string_view {
    return "picking";
}

auto PickingPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

void PickingPass::setup(const webgpu::Device& device) {
    auto shader_src = editor_resources::get_resource("editor/generated/shaders/picking.wgsl");
    PRECONDITION_MSG(shader_src,
                     "Missing embedded resource: editor/generated/shaders/picking.wgsl");

    auto shader = device.create_shader_module_from_source(*shader_src);

    uint32_t initial_capacity = 64;
    auto uniform_buffer = device.create_buffer(
        k_uniform_align * initial_capacity,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    // Create bind group layout: binding 0 = uniform (dynamic)
    WGPUBindGroupLayoutEntry bgl_entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    bgl_entry.binding = 0;
    bgl_entry.visibility =
        static_cast<WGPUShaderStage>(WGPUShaderStage_Vertex | WGPUShaderStage_Fragment);
    bgl_entry.buffer.type = WGPUBufferBindingType_Uniform;
    bgl_entry.buffer.hasDynamicOffset = true;
    bgl_entry.buffer.minBindingSize = sizeof(PickingUniforms);

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
                        .color_format(WGPUTextureFormat_R32Uint)
                        .depth_format(WGPUTextureFormat_Depth24Plus)
                        .depth_write(true)
                        .depth_compare(WGPUCompareFunction_Less)
                        .cull_mode(WGPUCullMode_Back)
                        .pipeline_layout(pipeline_layout)
                        .vertex_layout<editor_picking_shader::VertexLayout>()
                        .build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    m_state = Ready{
        std::move(shader), std::move(pipeline), std::move(uniform_buffer),
        bind_group,        bind_group_layout,   initial_capacity,
    };
}

void PickingPass::ensure_capacity(const webgpu::Device& device, uint32_t object_count) {
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

void PickingPass::add_to_frame_graph(rendering::FrameGraph& fg, const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    auto object_count = static_cast<uint32_t>(ctx.world.objects.size());
    if (object_count > 0) {
        ensure_capacity(ctx.device, object_count);
    }

    rendering::TextureDesc picking_desc;
    picking_desc.width = ctx.viewport_width;
    picking_desc.height = ctx.viewport_height;
    picking_desc.format = WGPUTextureFormat_R32Uint;
    picking_desc.usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc);
    picking_desc.clear_color = {static_cast<double>(UINT32_MAX), 0, 0, 0};

    rendering::TextureDesc depth_desc;
    depth_desc.width = ctx.viewport_width;
    depth_desc.height = ctx.viewport_height;
    depth_desc.format = WGPUTextureFormat_Depth24Plus;

    auto picking_ids = fg.find_or_create("picking_ids", picking_desc);
    auto depth = fg.find_or_create("picking_depth", depth_desc);

    auto queue = ctx.queue;
    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    auto* pipeline_handle = ready.pipeline.handle();
    auto uniform_buf = ready.uniform_buffer.handle();
    auto bind_group = ready.bind_group;
    const auto& world = ctx.world;

    for (uint32_t i = 0; i < object_count; ++i) {
        if (!world.objects[i].active) continue;
        const auto& obj = world.objects[i];
        PickingUniforms u{};
        u.mvp = proj_mat * view_mat * obj.transform;
        u.object_id = i;
        wgpuQueueWriteBuffer(queue, uniform_buf, i * k_uniform_align, &u, sizeof(u));
    }

    fg.add_pass("picking")
        .color(picking_ids)
        .depth(depth)
        .execute([=, &world](WGPURenderPassEncoder pass) {
            wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
            for (uint32_t i = 0; i < static_cast<uint32_t>(world.objects.size()); ++i) {
                if (!world.objects[i].active) continue;
                uint32_t dyn_offset = i * k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 1, &dyn_offset);
                const auto& mesh = world.meshes[world.objects[i].mesh_index];
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh.vertex_buffer.handle(), 0,
                                                     mesh.vertex_buffer.size());
                wgpuRenderPassEncoderSetIndexBuffer(pass, mesh.index_buffer.handle(),
                                                    WGPUIndexFormat_Uint32, 0,
                                                    mesh.index_buffer.size());
                wgpuRenderPassEncoderDrawIndexed(pass, mesh.index_count, 1, 0, 0, 0);
            }
        });
}
