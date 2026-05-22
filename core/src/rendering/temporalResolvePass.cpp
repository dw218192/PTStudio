#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/temporalResolvePass.h>
#include <core/rendering/temporalStorage.h>
#include <core/rendering/webgpu/device.h>
#include <imgui.h>
#include <temporal_resolve_shader_metadata.h>

#include <cstdint>
#include <glm/glm.hpp>

namespace pts::rendering {

// Must match TemporalResolveUniforms in shadow/temporal_resolve.slang.
struct TemporalResolveUniforms {
    glm::vec2 viewport_size;  // 0:  8
    float alpha;              // 8:  4
    float gamma;              // 12: 4 -> total 16
};
static_assert(sizeof(TemporalResolveUniforms) == 16,
              "TemporalResolveUniforms must match shader std140 layout");

TemporalResolvePass::Outputs TemporalResolvePass::add_to_frame_graph(
    FrameGraph& fg, const PassContext& ctx, const Inputs& in, TemporalStorageManager& storage) {
    PTS_ZONE_SCOPED;
    ensure_initialized(ctx.device);

    // No raw input -> nothing to resolve. Disabled -> pass the raw visibility
    // through unchanged so the consumer still gets a valid (untemporal)
    // texture; this is the A/B baseline for the variance clamp.
    if (!in.raw_visibility) {
        return {};
    }
    if (!m_enabled) {
        return {in.raw_visibility};
    }

    auto internal_bgl = fg.bind_group_layout(
        "temporal_resolve/internal",
        temporal_resolve_shader::create_bind_group_layout_0(ctx.device.handle()));

    auto* pipeline = fg.render_pipeline("temporal_resolve")
                         .shader("core/generated/shaders/shadow/temporal_resolve.wgsl")
                         .color_format(WGPUTextureFormat_R16Float)
                         .cull_mode(WGPUCullMode_None)
                         .bind_group_layouts({internal_bgl})
                         .build();

    // Reset the bootstrap counter if the viewport size changed -- the freshly
    // resized history texture has stale or undefined content for the new
    // dimensions, so the next frame must use alpha=1.
    if (m_history_width != ctx.viewport_width || m_history_height != ctx.viewport_height) {
        m_frame_counter = 0;
        m_history_width = ctx.viewport_width;
        m_history_height = ctx.viewport_height;
    }

    auto usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                               WGPUTextureUsage_TextureBinding);
    auto pp = storage.request_ping_pong(fg, "resolved_visibility", ctx.viewport_width,
                                        ctx.viewport_height, WGPUTextureFormat_R16Float, usage,
                                        m_frame_counter);

    BufferDesc uniform_buf_desc;
    uniform_buf_desc.size = sizeof(TemporalResolveUniforms);
    uniform_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_decl = create_buffer(fg, uniform_buf_desc, "uniforms");

    auto desc_decl = descriptor(fg, internal_bgl, "internal_desc")
                         .texture(0, in.raw_visibility)
                         .sampler(1, fg.sampler(WGPUSamplerBindingType_NonFiltering))
                         .texture(2, pp.read)
                         .sampler(3, fg.sampler(WGPUSamplerBindingType_Filtering))
                         .buffer(4, uniform_buf_decl, 0, sizeof(TemporalResolveUniforms))
                         .build();

    auto queue = ctx.queue;
    auto viewport_width = ctx.viewport_width;
    auto viewport_height = ctx.viewport_height;
    auto blend_weight = m_blend_weight;
    auto gamma = m_gamma;
    auto frame_counter = m_frame_counter;

    fg.add_pass("temporal_resolve")
        .read(in.raw_visibility)
        .read(pp.read)
        .color(pp.write)
        .execute([=](ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto uniform_buf = exec.get(uniform_buf_decl).buffer;
            auto desc = exec.get(desc_decl).bind_group;

            TemporalResolveUniforms uniforms{};
            uniforms.viewport_size = {static_cast<float>(viewport_width),
                                      static_cast<float>(viewport_height)};
            // First frame after reset: take the raw sample directly so we
            // don't blend into a zero-initialized history.
            uniforms.alpha = (frame_counter == 0) ? 1.0f : blend_weight;
            uniforms.gamma = gamma;
            wgpuQueueWriteBuffer(queue, uniform_buf, 0, &uniforms, sizeof(uniforms));

            wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, desc, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

    ++m_frame_counter;
    return {pp.write};
}

void TemporalResolvePass::draw_imgui() {
    ImGui::Checkbox("Enabled", &m_enabled);
    ImGui::SliderFloat("Blend Weight (curr)", &m_blend_weight, 0.001f, 1.0f, "%.3f");
    ImGui::SliderFloat("Variance Clamp Gamma", &m_gamma, 0.25f, 8.0f, "%.2f");
    ImGui::Text("Frame counter: %llu", static_cast<unsigned long long>(m_frame_counter));
    if (ImGui::Button("Reset history")) {
        reset_history();
    }
}

}  // namespace pts::rendering
