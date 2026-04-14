#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/gbufferPass.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/webgpu/device.h>
#include <gbuffer_shader_metadata.h>

#include <glm/glm.hpp>

namespace pts::rendering {

struct GBufferObjectUniforms {
    glm::mat4 mvp;
    glm::mat4 model_view;
};
static_assert(sizeof(GBufferObjectUniforms) == 128,
              "GBufferObjectUniforms must match shader std140 layout");

static constexpr IPass::DebugTarget k_debug_targets[] = {
    {"Normals", "scene_normals"},
};

auto GBufferPass::debug_targets() const noexcept -> std::pair<const DebugTarget*, uint32_t> {
    return {k_debug_targets, 1};
}

GBufferPass::Outputs GBufferPass::add_to_frame_graph(FrameGraph& fg, const PassContext& ctx,
                                                     const Inputs&) {
    PTS_ZONE_SCOPED;
    ensure_initialized(ctx.device);

    auto desc_layout = fg.bind_group_layout(
        "gbuffer/desc", gbuffer_shader::create_bind_group_layout_0(ctx.device.handle()));

    auto* pipeline_handle = fg.render_pipeline("gbuffer")
                                .shader("core/generated/shaders/gbuffer.wgsl")
                                .color_format(WGPUTextureFormat_RG16Float, 0)
                                .depth_format(WGPUTextureFormat_Depth32Float)
                                .depth_write(true)
                                .depth_compare(WGPUCompareFunction_LessEqual)
                                .cull_mode(WGPUCullMode_Back)
                                .bind_group_layouts({desc_layout})
                                .vertex_layout<gbuffer_shader::VertexLayout>()
                                .build();

    auto objects = ctx.world.get_objects();
    auto total_slots = static_cast<uint32_t>(objects.size());

    // Register per-object uniform buffer with frame graph
    uint64_t needed_size =
        std::max(uint64_t(1), static_cast<uint64_t>(total_slots)) * k_uniform_align;
    BufferDesc buf_desc;
    buf_desc.size = needed_size;
    buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_decl = create_buffer(fg, buf_desc, "uniforms");

    // Register descriptor with frame graph
    auto bg_decl = descriptor(fg, desc_layout, "bg0")
                       .buffer(0, uniform_buf_decl, 0, sizeof(GBufferObjectUniforms))
                       .build();

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

    auto depth_decl = create_texture(fg, depth_desc, "depth");
    auto normals_decl = create_texture(fg, normals_desc, "normals");

    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    auto queue = ctx.queue;
    const auto& world = ctx.world;

    fg.add_pass("gbuffer")
        .color(normals_decl)
        .depth(depth_decl)
        .execute([=, &world](ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto objs = world.get_objects();
            auto meshes = world.get_meshes();
            auto buf = exec.get(uniform_buf_decl).buffer;
            auto bg = exec.get(bg_decl).bind_group;

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

    return {depth_decl, normals_decl};
}

}  // namespace pts::rendering
