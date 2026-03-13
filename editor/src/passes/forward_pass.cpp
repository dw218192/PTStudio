#include "forward_pass.h"

#include <core/diagnostics.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <shader_metadata.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "editorResources.h"

using namespace pts;
using namespace pts::editor;

struct ForwardUniforms {
    glm::mat4 mvp;
    glm::mat4 model;
    glm::vec3 sun_dir;
    float time;
    uint32_t object_id;
    uint32_t _pad[3];
};
static_assert(sizeof(ForwardUniforms) == 160, "ForwardUniforms must match shader std140 layout");
static_assert(ForwardPass::kUniformAlign >= sizeof(ForwardUniforms),
              "Alignment must be >= uniform struct size");

ForwardPass::~ForwardPass() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group) {
            wgpuBindGroupRelease(ready->bind_group);
        }
        if (ready->bind_group_layout) {
            wgpuBindGroupLayoutRelease(ready->bind_group_layout);
        }
    }
}

auto ForwardPass::name() const noexcept -> std::string_view {
    return "forward";
}

auto ForwardPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

void ForwardPass::setup(const webgpu::Device& device) {
    auto shader_src = editor_resources::get_resource("generated/shaders/forward.wgsl");
    PRECONDITION_MSG(shader_src, "Missing embedded resource: generated/shaders/forward.wgsl");

    auto shader = device.create_shader_module_from_source(*shader_src);

    uint32_t initial_capacity = 64;
    auto uniform_buffer = device.create_buffer(
        kUniformAlign * initial_capacity,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    // Create bind group layout with hasDynamicOffset for per-object uniforms
    WGPUBindGroupLayoutEntry entry0 = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entry0.binding = 0;
    entry0.visibility =
        static_cast<WGPUShaderStage>(WGPUShaderStage_Vertex | WGPUShaderStage_Fragment);
    entry0.buffer.type = WGPUBufferBindingType_Uniform;
    entry0.buffer.hasDynamicOffset = true;
    entry0.buffer.minBindingSize = sizeof(ForwardUniforms);

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = 1;
    bgl_desc.entries = &entry0;
    auto bind_group_layout = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);

    // Bind group: offset=0, size=sizeof(ForwardUniforms); dynamic offset applied per draw
    WGPUBindGroupEntry bg_entry = WGPU_BIND_GROUP_ENTRY_INIT;
    bg_entry.binding = 0;
    bg_entry.buffer = uniform_buffer.handle();
    bg_entry.offset = 0;
    bg_entry.size = sizeof(ForwardUniforms);

    WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg_desc.layout = bind_group_layout;
    bg_desc.entryCount = 1;
    bg_desc.entries = &bg_entry;
    auto bind_group = wgpuDeviceCreateBindGroup(device.handle(), &bg_desc);

    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &bind_group_layout;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

    auto pipeline = webgpu::RenderPipelineBuilder(device)
                        .shader(shader)
                        .color_format(WGPUTextureFormat_RGBA8Unorm)
                        .color_format(WGPUTextureFormat_R32Uint, 1)
                        .depth_format(WGPUTextureFormat_Depth24Plus)
                        .depth_write(true)
                        .depth_compare(WGPUCompareFunction_Less)
                        .cull_mode(WGPUCullMode_Back)
                        .pipeline_layout(pipeline_layout)
                        .vertex_layout<editor_shader::VertexLayout>()
                        .build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    m_state = Ready{
        std::move(shader), std::move(pipeline), std::move(uniform_buffer),
        bind_group,        bind_group_layout,   initial_capacity,
    };
}

void ForwardPass::ensure_capacity(const webgpu::Device& device, uint32_t object_count) {
    auto& ready = std::get<Ready>(m_state);
    if (object_count <= ready.capacity) return;

    uint32_t new_capacity = ready.capacity;
    while (new_capacity < object_count) {
        new_capacity *= 2;
    }

    ready.uniform_buffer = device.create_buffer(
        kUniformAlign * new_capacity,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    // Recreate bind group pointing to new buffer
    if (ready.bind_group) {
        wgpuBindGroupRelease(ready.bind_group);
    }

    WGPUBindGroupEntry bg_entry = WGPU_BIND_GROUP_ENTRY_INIT;
    bg_entry.binding = 0;
    bg_entry.buffer = ready.uniform_buffer.handle();
    bg_entry.offset = 0;
    bg_entry.size = sizeof(ForwardUniforms);

    WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg_desc.layout = ready.bind_group_layout;
    bg_desc.entryCount = 1;
    bg_desc.entries = &bg_entry;
    ready.bind_group = wgpuDeviceCreateBindGroup(device.handle(), &bg_desc);
    ready.capacity = new_capacity;
}

void ForwardPass::add_to_frame_graph(rendering::FrameGraph& fg, const rendering::PassContext& ctx) {
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    auto object_count = static_cast<uint32_t>(ctx.world.objects.size());
    if (object_count > 0) {
        ensure_capacity(ctx.device, object_count);
    }

    rendering::TextureDesc color_desc;
    color_desc.width = ctx.viewport_width;
    color_desc.height = ctx.viewport_height;
    color_desc.format = WGPUTextureFormat_RGBA8Unorm;
    color_desc.clear_color = {0.15, 0.15, 0.18, 1.0};

    rendering::TextureDesc picking_desc;
    picking_desc.width = ctx.viewport_width;
    picking_desc.height = ctx.viewport_height;
    picking_desc.format = WGPUTextureFormat_R32Uint;
    picking_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                                       WGPUTextureUsage_CopySrc);
    picking_desc.clear_color = {static_cast<double>(UINT32_MAX), 0, 0, 0};

    rendering::TextureDesc depth_desc;
    depth_desc.width = ctx.viewport_width;
    depth_desc.height = ctx.viewport_height;
    depth_desc.format = WGPUTextureFormat_Depth24Plus;

    auto color = fg.find_or_create("scene_color", color_desc);
    auto picking_ids = fg.find_or_create("picking_ids", picking_desc);
    auto depth = fg.find_or_create("scene_depth", depth_desc);

    auto queue = ctx.queue;
    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    auto elapsed_time = ctx.time;
    auto* pipeline_handle = ready.pipeline.handle();
    auto uniform_buf = ready.uniform_buffer.handle();
    auto bind_group = ready.bind_group;
    const auto& world = ctx.world;

    // Write all per-object uniforms before encoding the render pass.
    // Each object gets its own aligned slice so dynamic offsets work correctly.
    for (uint32_t i = 0; i < object_count; ++i) {
        const auto& obj = world.objects[i];
        ForwardUniforms u{};
        u.mvp = proj_mat * view_mat * obj.transform;
        u.model = obj.transform;
        u.sun_dir = glm::normalize(glm::vec3(0.3f, 1.0f, 0.5f));
        u.time = elapsed_time;
        u.object_id = i;
        wgpuQueueWriteBuffer(queue, uniform_buf, i * kUniformAlign, &u, sizeof(u));
    }

    fg.add_pass("forward").color(color).color(picking_ids).depth(depth).execute(
        [=, &world](WGPURenderPassEncoder pass) {
            wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
            for (uint32_t i = 0; i < static_cast<uint32_t>(world.objects.size()); ++i) {
                uint32_t dyn_offset = i * kUniformAlign;
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
