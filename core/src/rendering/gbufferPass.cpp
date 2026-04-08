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

GBufferPass::GBufferPass(const ShaderLoader& sl) : IPass(sl) {
}

GBufferPass::~GBufferPass() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bgl) wgpuBindGroupLayoutRelease(ready->bgl);
    }
}

auto GBufferPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

static constexpr IPass::DebugTarget k_debug_targets[] = {
    {"Normals", "scene_normals"},
};

auto GBufferPass::debug_targets() const noexcept -> std::pair<const DebugTarget*, uint32_t> {
    return {k_debug_targets, 1};
}

void GBufferPass::do_setup(const webgpu::Device& device) {
    // Release existing state for re-entry (hot-reload)
    if (auto* ready = std::get_if<Ready>(&m_state)) {
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

    m_state = Ready{
        std::move(shader),
        std::move(pipeline),
        bgl,
    };
}

GBufferPass::Outputs GBufferPass::add_to_frame_graph(FrameGraph& fg, const PassContext& ctx,
                                                     const Inputs&) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    auto objects = ctx.world.get_objects();
    auto total_slots = static_cast<uint32_t>(objects.size());

    // Register per-object uniform buffer with frame graph
    uint64_t needed_size =
        std::max(uint64_t(1), static_cast<uint64_t>(total_slots)) * k_uniform_align;
    BufferDesc buf_desc;
    buf_desc.size = needed_size;
    buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_handle = create_buffer(fg, buf_desc, "uniforms");

    // Register bind group with frame graph
    BindGroupDesc bg_desc;
    bg_desc.layout = ready.bgl;
    bg_desc.entries = {
        {0, ManagedBufferBinding{uniform_buf_handle, 0, sizeof(GBufferObjectUniforms)}}};
    auto bg_handle = create_bind_group(fg, std::move(bg_desc), "bg0");

    // Create/find frame graph texture resources
    TextureDesc depth_desc;
    depth_desc.width = ctx.viewport_width;
    depth_desc.height = ctx.viewport_height;
    depth_desc.format = WGPUTextureFormat_Depth32Float;

    TextureDesc normals_desc;
    normals_desc.width = ctx.viewport_width;
    normals_desc.height = ctx.viewport_height;
    normals_desc.format = WGPUTextureFormat_RG16Float;
    normals_desc.clear_color = {0, 0, 0, 0};

    auto depth = create_texture(fg, depth_desc, "depth");
    auto normals = create_texture(fg, normals_desc, "normals");

    auto* pipeline_handle = ready.pipeline.handle();
    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    auto queue = ctx.queue;
    const auto& world = ctx.world;

    fg.add_pass("gbuffer").color(normals).depth(depth).execute(
        [=, &fg, &world](WGPURenderPassEncoder pass) {
            auto objs = world.get_objects();
            auto meshes = world.get_meshes();
            auto buf = fg.get_buffer_ref(uniform_buf_handle).handle();
            auto bg = fg.get_bind_group_ref(bg_handle).handle();

            // Upload per-object uniforms
            {
                PTS_ZONE_NAMED("gbuffer uniform upload");
                for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                    if (!objs[i].active()) continue;
                    if (!objs[i]->visible) continue;
                    GBufferObjectUniforms u{};
                    u.mvp = proj_mat * view_mat * objs[i]->transform;
                    u.model_view = view_mat * objs[i]->transform;
                    wgpuQueueWriteBuffer(queue, buf, i * k_uniform_align, &u, sizeof(u));
                }
            }

            wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
            for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                if (!objs[i].active()) continue;
                if (!objs[i]->visible) continue;
                uint32_t dyn_offset = i * k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 1, &dyn_offset);
                const auto& mesh = meshes[objs[i]->mesh_index];
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh->vertex_buffer.handle(), 0,
                                                     mesh->vertex_buffer.size());
                wgpuRenderPassEncoderSetIndexBuffer(pass, mesh->index_buffer.handle(),
                                                    WGPUIndexFormat_Uint32, 0,
                                                    mesh->index_buffer.size());
                wgpuRenderPassEncoderDrawIndexed(pass, mesh->index_count, 1, 0, 0, 0);
            }
        });

    return {depth, normals};
}

}  // namespace pts::rendering
