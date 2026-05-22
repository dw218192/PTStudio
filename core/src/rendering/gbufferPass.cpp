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
#include <vector>

namespace pts::rendering {

struct GBufferObjectUniforms {
    glm::mat4 mvp;
    glm::mat4 model_view;
    glm::mat4 prev_mvp;
};
static_assert(sizeof(GBufferObjectUniforms) == 192,
              "GBufferObjectUniforms must match shader std140 layout");

static constexpr IPass::DebugTarget k_debug_targets[] = {
    {"Normals", "scene_normals"},
    {"Motion", "scene_motion"},
};

auto GBufferPass::debug_targets() const noexcept -> std::pair<const DebugTarget*, uint32_t> {
    return {k_debug_targets, 2};
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
                                .color_format(WGPUTextureFormat_RG16Float, 1)
                                .depth_format(WGPUTextureFormat_Depth32Float)
                                .depth_write(true)
                                .depth_compare(WGPUCompareFunction_LessEqual)
                                .cull_mode(WGPUCullMode_Back)
                                .bind_group_layouts({desc_layout})
                                .vertex_layout<gbuffer_shader::VertexLayout>()
                                .build();

    auto objects = ctx.world.get_objects().span_raw();
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
    auto desc_decl = descriptor(fg, desc_layout, "desc0")
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

    TextureDesc motion_desc;
    motion_desc.width = ctx.viewport_width;
    motion_desc.height = ctx.viewport_height;
    motion_desc.format = WGPUTextureFormat_RG16Float;
    // Skybox / cleared pixels have no surface motion -- clear to zero so
    // consumers that sample (uv + motion) read back the same uv.
    motion_desc.clear_color = {0, 0, 0, 0};

    auto depth_decl = create_texture(fg, depth_desc, "depth");
    auto normals_decl = create_texture(fg, normals_desc, "normals");
    auto motion_decl = create_texture(fg, motion_desc, "motion");

    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;

    // Build per-slot prev_mvp using last frame's transforms + view/proj. New
    // / re-activated slots fall back to current transforms (motion = 0) so
    // there's no stale-state artifact on the first frame they appear.
    glm::mat4 prev_view = m_prev_camera_valid ? m_prev_view : view_mat;
    glm::mat4 prev_proj = m_prev_camera_valid ? m_prev_proj : proj_mat;
    std::vector<glm::mat4> prev_mvps(total_slots, glm::mat4(1.0f));
    for (uint32_t i = 0; i < total_slots; ++i) {
        if (!objects[i].active) continue;
        if (!objects[i].value.visible) continue;
        const bool prev_known =
            m_prev_camera_valid && i < m_prev_objects.size() && m_prev_objects[i].valid;
        glm::mat4 prev_xform =
            prev_known ? m_prev_objects[i].transform : objects[i].value.transform;
        prev_mvps[i] = prev_proj * prev_view * prev_xform;
    }

    // Snapshot current state for next frame. Inactive / invisible slots get
    // valid = false so they're treated as "new" if they turn on later.
    m_prev_objects.assign(total_slots, PrevObjectState{});
    for (uint32_t i = 0; i < total_slots; ++i) {
        if (!objects[i].active) continue;
        if (!objects[i].value.visible) continue;
        m_prev_objects[i].transform = objects[i].value.transform;
        m_prev_objects[i].valid = true;
    }
    m_prev_view = view_mat;
    m_prev_proj = proj_mat;
    m_prev_camera_valid = true;

    auto queue = ctx.queue;
    const auto& world = ctx.world;

    fg.add_pass("gbuffer")
        .color(normals_decl)
        .color(motion_decl)
        .depth(depth_decl)
        .execute([=, &world](ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto objs = world.get_objects().span_raw();
            auto meshes = world.get_meshes().span_raw();
            auto buf = exec.get(uniform_buf_decl).buffer;
            auto desc = exec.get(desc_decl).bind_group;

            // Upload per-object uniforms
            {
                PTS_ZONE_NAMED("gbuffer uniform upload");
                for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                    if (!objs[i].active) continue;
                    if (!objs[i].value.visible) continue;
                    GBufferObjectUniforms u{};
                    u.mvp = proj_mat * view_mat * objs[i].value.transform;
                    u.model_view = view_mat * objs[i].value.transform;
                    u.prev_mvp = prev_mvps[i];
                    wgpuQueueWriteBuffer(queue, buf, i * k_uniform_align, &u, sizeof(u));
                }
            }

            wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
            for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                if (!objs[i].active) continue;
                if (!objs[i].value.visible) continue;
                uint32_t dyn_offset = i * k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, desc, 1, &dyn_offset);
                const auto& mesh = meshes[objs[i].value.mesh_index].value;
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh.vertex_buffer.handle(), 0,
                                                     mesh.vertex_buffer.size());
                wgpuRenderPassEncoderSetIndexBuffer(pass, mesh.index_buffer.handle(),
                                                    WGPUIndexFormat_Uint32, 0,
                                                    mesh.index_buffer.size());
                wgpuRenderPassEncoderDrawIndexed(pass, mesh.index_count, 1, 0, 0, 0);
            }
        });

    return {depth_decl, normals_decl, motion_decl};
}

}  // namespace pts::rendering
