#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/shadowVisibilityPass.h>
#include <core/rendering/webgpu/device.h>
#include <imgui.h>
#include <shadow_visibility_shader_metadata.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

namespace pts::rendering {

// Must match ShadowVisibilityUniforms in shadow/shadow_visibility.slang.
struct ShadowVisibilityUniforms {
    glm::mat4 inv_view_proj;      // 0:  64
    glm::vec2 viewport_size;      // 64: 8
    uint32_t shadow_light_index;  // 72: 4
    uint32_t frame_index;         // 76: 4 -> total 80
};
static_assert(sizeof(ShadowVisibilityUniforms) == 80,
              "ShadowVisibilityUniforms must match shader std140 layout");

ShadowVisibilityPass::Outputs ShadowVisibilityPass::add_to_frame_graph(FrameGraph& fg,
                                                                       const PassContext& ctx,
                                                                       const Inputs& in) {
    PTS_ZONE_SCOPED;
    ensure_initialized(ctx.device);

    if (!m_enabled || in.shadow_light_index == UINT32_MAX || !in.depth || !in.shadow_array ||
        !in.shadow_info) {
        return {};
    }

    auto internal_bgl = fg.bind_group_layout(
        "shadow_visibility/internal",
        shadow_visibility_shader::create_bind_group_layout_0(ctx.device.handle()));

    auto* pipeline = fg.render_pipeline("shadow_visibility")
                         .shader("core/generated/shaders/shadow/shadow_visibility.wgsl")
                         .color_format(WGPUTextureFormat_R16Float)
                         .cull_mode(WGPUCullMode_None)
                         .bind_group_layouts({internal_bgl})
                         .build();

    TextureDesc vis_desc;
    vis_desc.width = ctx.viewport_width;
    vis_desc.height = ctx.viewport_height;
    vis_desc.format = WGPUTextureFormat_R16Float;
    vis_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                                   WGPUTextureUsage_TextureBinding);
    vis_desc.clear_color = {1, 1, 1, 1};
    auto vis_decl = create_texture(fg, vis_desc, "raw_visibility");

    BufferDesc uniform_buf_desc;
    uniform_buf_desc.size = sizeof(ShadowVisibilityUniforms);
    uniform_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_decl = create_buffer(fg, uniform_buf_desc, "uniforms");

    auto desc_decl = descriptor(fg, internal_bgl, "internal_desc")
                         .texture(0, in.depth)
                         .sampler(1, fg.sampler(WGPUSamplerBindingType_NonFiltering))
                         .buffer(2, in.shadow_info)
                         .texture(3, in.shadow_array)
                         .sampler(4, fg.sampler(WGPUSamplerBindingType_NonFiltering))
                         .buffer(5, uniform_buf_decl, 0, sizeof(ShadowVisibilityUniforms))
                         .build();

    auto queue = ctx.queue;
    auto view_matrix = ctx.view_matrix;
    auto proj_matrix = ctx.proj_matrix;
    auto viewport_width = ctx.viewport_width;
    auto viewport_height = ctx.viewport_height;
    auto shadow_light_index = in.shadow_light_index;
    auto frame_counter = m_frame_counter;

    fg.add_pass("shadow_visibility")
        .read(in.depth)
        .read(in.shadow_array)
        .color(vis_decl)
        .execute([=](ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto uniform_buf = exec.get(uniform_buf_decl).buffer;
            auto desc = exec.get(desc_decl).bind_group;

            ShadowVisibilityUniforms uniforms{};
            uniforms.inv_view_proj = glm::inverse(proj_matrix * view_matrix);
            uniforms.viewport_size = {static_cast<float>(viewport_width),
                                      static_cast<float>(viewport_height)};
            uniforms.shadow_light_index = shadow_light_index;
            uniforms.frame_index = static_cast<uint32_t>(frame_counter & 0xffffffffull);
            wgpuQueueWriteBuffer(queue, uniform_buf, 0, &uniforms, sizeof(uniforms));

            wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, desc, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

    ++m_frame_counter;
    return {vis_decl};
}

void ShadowVisibilityPass::draw_imgui() {
    ImGui::Checkbox("Enabled", &m_enabled);
}

}  // namespace pts::rendering
