#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/temporalStorage.h>
#include <core/rendering/temporalVisibilityPass.h>
#include <core/rendering/webgpu/device.h>
#include <imgui.h>
#include <shadow_temporal_shader_metadata.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

namespace pts::rendering {

// Must match TemporalVisibilityUniforms in shadow/temporal.slang.
struct TemporalVisibilityUniforms {
    glm::mat4 inv_view_proj;      // 0:  64
    glm::vec2 viewport_size;      // 64: 8
    float alpha;                  // 72: 4
    uint32_t shadow_light_index;  // 76: 4
    uint32_t frame_index;         // 80: 4
    uint32_t _pad[3];             // 84: 12 -> total 96
};
static_assert(sizeof(TemporalVisibilityUniforms) == 96,
              "TemporalVisibilityUniforms must match shader std140 layout");

TemporalVisibilityPass::Outputs TemporalVisibilityPass::add_to_frame_graph(
    FrameGraph& fg, const PassContext& ctx, const Inputs& in, TemporalStorageManager& storage) {
    PTS_ZONE_SCOPED;
    ensure_initialized(ctx.device);

    if (!m_enabled || in.shadow_light_index == UINT32_MAX || !in.depth || !in.shadow_array ||
        !in.shadow_info) {
        return {};
    }

    auto internal_bgl = fg.bind_group_layout(
        "temporal_visibility/internal",
        shadow_temporal_shader::create_bind_group_layout_0(ctx.device.handle()));

    auto* pipeline = fg.render_pipeline("temporal_visibility")
                         .shader("core/generated/shaders/shadow/temporal.wgsl")
                         .color_format(WGPUTextureFormat_R16Float)
                         .cull_mode(WGPUCullMode_None)
                         .bind_group_layouts({internal_bgl})
                         .build();

    // Reset bootstrap counter if the viewport size changed -- the freshly
    // resized history texture has stale or undefined content for the new
    // dimensions, so the next frame must use alpha=1.
    if (m_history_width != ctx.viewport_width || m_history_height != ctx.viewport_height) {
        m_frame_counter = 0;
        m_history_width = ctx.viewport_width;
        m_history_height = ctx.viewport_height;
    }

    auto usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                               WGPUTextureUsage_TextureBinding);
    auto pp =
        storage.request_ping_pong(fg, "shadow_visibility", ctx.viewport_width, ctx.viewport_height,
                                  WGPUTextureFormat_R16Float, usage, m_frame_counter);

    BufferDesc uniform_buf_desc;
    uniform_buf_desc.size = sizeof(TemporalVisibilityUniforms);
    uniform_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_decl = create_buffer(fg, uniform_buf_desc, "uniforms");

    auto desc_decl = descriptor(fg, internal_bgl, "internal_desc")
                         .texture(0, in.depth)
                         .sampler(1, fg.sampler(WGPUSamplerBindingType_NonFiltering))
                         .buffer(2, in.shadow_info)
                         .texture(3, in.shadow_array)
                         .sampler(4, fg.sampler(WGPUSamplerBindingType_NonFiltering))
                         .texture(5, pp.read)
                         .sampler(6, fg.sampler(WGPUSamplerBindingType_Filtering))
                         .buffer(7, uniform_buf_decl, 0, sizeof(TemporalVisibilityUniforms))
                         .build();

    auto queue = ctx.queue;
    auto view_matrix = ctx.view_matrix;
    auto proj_matrix = ctx.proj_matrix;
    auto viewport_width = ctx.viewport_width;
    auto viewport_height = ctx.viewport_height;
    auto blend_weight = m_blend_weight;
    auto shadow_light_index = in.shadow_light_index;
    auto frame_counter = m_frame_counter;

    fg.add_pass("temporal_visibility")
        .read(in.depth)
        .read(in.shadow_array)
        .read(pp.read)
        .color(pp.write)
        .execute([=](ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto uniform_buf = exec.get(uniform_buf_decl).buffer;
            auto desc = exec.get(desc_decl).bind_group;

            TemporalVisibilityUniforms uniforms{};
            uniforms.inv_view_proj = glm::inverse(proj_matrix * view_matrix);
            uniforms.viewport_size = {static_cast<float>(viewport_width),
                                      static_cast<float>(viewport_height)};
            // First frame after reset: take the curr sample directly so we
            // don't blend into a zero-initialized history.
            uniforms.alpha = (frame_counter == 0) ? 1.0f : blend_weight;
            uniforms.shadow_light_index = shadow_light_index;
            uniforms.frame_index = static_cast<uint32_t>(frame_counter & 0xffffffffull);
            wgpuQueueWriteBuffer(queue, uniform_buf, 0, &uniforms, sizeof(uniforms));

            wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, desc, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

    ++m_frame_counter;
    return {pp.write};
}

void TemporalVisibilityPass::draw_imgui() {
    ImGui::Checkbox("Enabled", &m_enabled);
    ImGui::SliderFloat("Blend Weight (curr)", &m_blend_weight, 0.001f, 1.0f, "%.3f");
    ImGui::Text("Frame counter: %llu", static_cast<unsigned long long>(m_frame_counter));
    if (ImGui::Button("Reset history")) {
        reset_history();
    }
}

}  // namespace pts::rendering
