#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/gbufferPass.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <gbuffer_shader_metadata.h>

#include <glm/glm.hpp>

namespace pts::rendering {

struct GBufferObjectUniforms {
    glm::mat4 mvp;
    glm::mat4 model_view;
};
static_assert(sizeof(GBufferObjectUniforms) == 128,
              "GBufferObjectUniforms must match shader std140 layout");

GBufferPass::GBufferPass(const ShaderLoader& sl) : IRenderPass(sl) {
}

GBufferPass::~GBufferPass() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group) wgpuBindGroupRelease(ready->bind_group);
        if (ready->bgl) wgpuBindGroupLayoutRelease(ready->bgl);
    }
}

auto GBufferPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

void GBufferPass::do_setup(const webgpu::Device& device) {
    // Release existing state for re-entry (hot-reload)
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group) wgpuBindGroupRelease(ready->bind_group);
        if (ready->bgl) wgpuBindGroupLayoutRelease(ready->bgl);
    }

    auto shader_src = get_shader_loader().load("core/generated/shaders/gbuffer.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    // BGL: binding 0 = dynamic uniform buffer (two mat4 = 128 bytes)
    WGPUBindGroupLayoutEntry bgl_entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    bgl_entry.binding = 0;
    bgl_entry.visibility =
        static_cast<WGPUShaderStage>(WGPUShaderStage_Vertex | WGPUShaderStage_Fragment);
    bgl_entry.buffer.type = WGPUBufferBindingType_Uniform;
    bgl_entry.buffer.hasDynamicOffset = true;
    bgl_entry.buffer.minBindingSize = sizeof(GBufferObjectUniforms);

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = 1;
    bgl_desc.entries = &bgl_entry;
    auto bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);

    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &bgl;
    auto pipeline_layout = wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

    auto pipeline = webgpu::RenderPipelineBuilder(device)
                        .shader(shader)
                        .color_format(WGPUTextureFormat_RG16Float, 0)
                        .depth_format(WGPUTextureFormat_Depth32Float)
                        .depth_write(true)
                        .depth_compare(WGPUCompareFunction_LessEqual)
                        .cull_mode(WGPUCullMode_Back)
                        .pipeline_layout(pipeline_layout)
                        .vertex_layout<gbuffer_shader::VertexLayout>()
                        .build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    uint32_t initial_capacity = 64;
    auto uniform_buf = device.create_buffer(
        k_uniform_align * initial_capacity,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    // Create bind group
    WGPUBindGroupEntry bg_entry = WGPU_BIND_GROUP_ENTRY_INIT;
    bg_entry.binding = 0;
    bg_entry.buffer = uniform_buf.handle();
    bg_entry.offset = 0;
    bg_entry.size = sizeof(GBufferObjectUniforms);

    WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg_desc.layout = bgl;
    bg_desc.entryCount = 1;
    bg_desc.entries = &bg_entry;
    auto bind_group = wgpuDeviceCreateBindGroup(device.handle(), &bg_desc);

    m_state = Ready{
        std::move(shader), std::move(pipeline), std::move(uniform_buf), bgl,
        bind_group,        initial_capacity,
    };
}

void GBufferPass::add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    auto objects = ctx.world.get_objects();
    auto total_slots = static_cast<uint32_t>(objects.size());

    // Resize uniform buffer if needed
    if (total_slots > 0 && total_slots > ready.object_capacity) {
        uint32_t new_capacity = ready.object_capacity;
        while (new_capacity < total_slots) {
            new_capacity *= 2;
        }

        ready.per_object_uniform_buf = ctx.device.create_buffer(
            static_cast<uint64_t>(new_capacity) * k_uniform_align,
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));
        ready.object_capacity = new_capacity;

        // Recreate bind group
        if (ready.bind_group) wgpuBindGroupRelease(ready.bind_group);

        WGPUBindGroupEntry bg_entry = WGPU_BIND_GROUP_ENTRY_INIT;
        bg_entry.binding = 0;
        bg_entry.buffer = ready.per_object_uniform_buf.handle();
        bg_entry.offset = 0;
        bg_entry.size = sizeof(GBufferObjectUniforms);

        WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bg_desc.layout = ready.bgl;
        bg_desc.entryCount = 1;
        bg_desc.entries = &bg_entry;
        ready.bind_group = wgpuDeviceCreateBindGroup(ctx.device.handle(), &bg_desc);
    }

    // Upload per-object uniforms
    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    auto uniform_buf = ready.per_object_uniform_buf.handle();

    {
        PTS_ZONE_NAMED("gbuffer uniform upload");
        for (uint32_t i = 0; i < total_slots; ++i) {
            if (!objects[i].active()) continue;
            if (!objects[i]->visible) continue;
            GBufferObjectUniforms u{};
            u.mvp = proj_mat * view_mat * objects[i]->transform;
            u.model_view = view_mat * objects[i]->transform;
            wgpuQueueWriteBuffer(ctx.queue, uniform_buf, i * k_uniform_align, &u, sizeof(u));
        }
    }

    // Create/find frame graph resources
    TextureDesc depth_desc;
    depth_desc.width = ctx.viewport_width;
    depth_desc.height = ctx.viewport_height;
    depth_desc.format = WGPUTextureFormat_Depth32Float;

    TextureDesc normals_desc;
    normals_desc.width = ctx.viewport_width;
    normals_desc.height = ctx.viewport_height;
    normals_desc.format = WGPUTextureFormat_RG16Float;
    normals_desc.clear_color = {0, 0, 0, 0};

    auto depth = fg.find_or_create("scene_depth", depth_desc);
    auto normals = fg.find_or_create("scene_normals", normals_desc);

    auto* pipeline_handle = ready.pipeline.handle();
    auto bind_group = ready.bind_group;
    const auto& world = ctx.world;

    fg.add_pass("gbuffer").color(normals).depth(depth).execute(
        [=, &world](WGPURenderPassEncoder pass) {
            auto objs = world.get_objects();
            auto meshes = world.get_meshes();

            wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
            for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                if (!objs[i].active()) continue;
                if (!objs[i]->visible) continue;
                uint32_t dyn_offset = i * k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 1, &dyn_offset);
                const auto& mesh = meshes[objs[i]->mesh_index];
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh->vertex_buffer.handle(), 0,
                                                     mesh->vertex_buffer.size());
                wgpuRenderPassEncoderSetIndexBuffer(pass, mesh->index_buffer.handle(),
                                                    WGPUIndexFormat_Uint32, 0,
                                                    mesh->index_buffer.size());
                wgpuRenderPassEncoderDrawIndexed(pass, mesh->index_count, 1, 0, 0, 0);
            }
        });
}

}  // namespace pts::rendering
