#include <contact_shadow_shader_metadata.h>
#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/contactShadowPass.h>
#include <core/rendering/fallbackPool.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/gbufferPass.h>
#include <core/rendering/passContext.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/webgpu/device.h>
#include <imgui.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

namespace pts::rendering {

// Must match ContactShadowUniforms in contact_shadow.slang (std140 layout).
struct ContactShadowUniforms {
    glm::mat4 projection;      // 0:   64
    glm::mat4 inv_projection;  // 64:  64
    glm::mat4 view;            // 128: 64
    glm::vec2 viewport_size;   // 192: 8
    float max_distance;        // 200: 4
    float thickness;           // 204: 4
    float normal_offset;       // 208: 4
    int32_t step_count;        // 212: 4
    uint32_t light_count;      // 216: 4
    uint32_t _pad;             // 220: 4 → total 224
};
static_assert(sizeof(ContactShadowUniforms) == 224,
              "ContactShadowUniforms must match shader std140 layout");

static constexpr IPass::DebugTarget k_debug_targets[] = {
    {"Contact Shadow", "contact_shadow"},
};

auto ContactShadowPass::debug_targets() const noexcept -> std::pair<const DebugTarget*, uint32_t> {
    return {k_debug_targets, m_enabled ? 1u : 0u};
}

ContactShadowPass::Outputs ContactShadowPass::add_to_frame_graph(FrameGraph& fg,
                                                                 const PassContext& ctx,
                                                                 const Inputs& in,
                                                                 FallbackPool& fallbacks) {
    PTS_ZONE_SCOPED;
    ensure_initialized(ctx.device);

    // Consumer layout registered up-front by the owning renderer (forwardPass)
    // from its shader's reflection; the consumer-side bind group shape is a
    // property of the downstream consumer, not of contact_shadow.slang.
    auto consumer_bgl = fg.bind_group_layout("contact_shadow/consumer");

    if (!m_enabled) {
        auto fallback_view = fallbacks.view(WGPUTextureFormat_R8Unorm, WGPUTextureViewDimension_2D);
        auto consumer = descriptor(fg, consumer_bgl, "consumer_desc")
                            .external_view(0, fallback_view)
                            .sampler(1, fg.sampler(WGPUSamplerBindingType_Filtering))
                            .build();
        return {{}, consumer};
    }

    auto internal_bgl = fg.bind_group_layout(
        "contact_shadow/internal",
        contact_shadow_shader::create_bind_group_layout_0(ctx.device.handle()));

    auto* pipeline = fg.render_pipeline("contact_shadow")
                         .shader("core/generated/shaders/contact_shadow.wgsl")
                         .color_format(WGPUTextureFormat_R8Unorm)
                         .cull_mode(WGPUCullMode_None)
                         .bind_group_layouts({internal_bgl})
                         .build();

    // ── Frame graph resources ──
    TextureDesc cs_desc;
    cs_desc.width = ctx.viewport_width;
    cs_desc.height = ctx.viewport_height;
    cs_desc.format = WGPUTextureFormat_R8Unorm;
    cs_desc.clear_color = {1, 1, 1, 1};
    auto cs_decl = create_texture(fg, cs_desc, "contact_shadow");

    BufferDesc uniform_buf_desc;
    uniform_buf_desc.size = sizeof(ContactShadowUniforms);
    uniform_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_decl = create_buffer(fg, uniform_buf_desc, "cs_uniforms");

    // Internal descriptor: depth(0), depth_sampler(1), normals(2), normals_sampler(3),
    //                      uniforms(4), lights(5)
    auto bg_decl = descriptor(fg, internal_bgl, "cs_bg")
                       .texture(0, in.depth)
                       .sampler(1, fg.sampler(WGPUSamplerBindingType_NonFiltering))
                       .texture(2, in.normals)
                       .sampler(3, fg.sampler(WGPUSamplerBindingType_Filtering))
                       .buffer(4, uniform_buf_decl, 0, sizeof(ContactShadowUniforms))
                       .external_buffer(5, in.light_buffer, 0, WGPU_WHOLE_SIZE)
                       .build();

    // Consumer descriptor: managed CS texture + sampler
    auto consumer = descriptor(fg, consumer_bgl, "consumer_desc")
                        .texture(0, cs_decl)
                        .sampler(1, fg.sampler(WGPUSamplerBindingType_Filtering))
                        .build();

    // Capture scalars for lambda
    auto queue = ctx.queue;
    auto proj_matrix = ctx.proj_matrix;
    auto view_matrix = ctx.view_matrix;
    auto viewport_width = ctx.viewport_width;
    auto viewport_height = ctx.viewport_height;
    auto light_count = ctx.world.gpu_light_count();
    auto max_distance = m_max_distance;
    auto thickness = m_thickness;
    auto normal_offset = m_normal_offset;
    auto step_count = m_step_count;

    fg.add_pass("contact_shadow_gen")
        .read(in.depth)
        .read(in.normals)
        .color(cs_decl)
        .execute([=](ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto uniform_buf = exec.get(uniform_buf_decl).buffer;
            auto bg = exec.get(bg_decl).bind_group;

            ContactShadowUniforms uniforms{};
            uniforms.projection = proj_matrix;
            uniforms.inv_projection = glm::inverse(proj_matrix);
            uniforms.view = view_matrix;
            uniforms.viewport_size = {
                static_cast<float>(viewport_width),
                static_cast<float>(viewport_height),
            };
            uniforms.max_distance = max_distance;
            uniforms.thickness = thickness;
            uniforms.normal_offset = normal_offset;
            uniforms.step_count = step_count;
            uniforms.light_count = light_count;
            wgpuQueueWriteBuffer(queue, uniform_buf, 0, &uniforms, sizeof(uniforms));

            wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

    return {cs_decl, consumer};
}

void ContactShadowPass::draw_imgui() {
    ImGui::Checkbox("Enabled", &m_enabled);
    ImGui::SliderFloat("Max Distance", &m_max_distance, 0.01f, 2.0f);
    ImGui::SliderFloat("Thickness", &m_thickness, 0.001f, 0.2f);
    ImGui::SliderFloat("Normal Offset", &m_normal_offset, 0.0f, 0.1f);
    ImGui::SliderInt("Step Count", &m_step_count, 4, 64);
}

}  // namespace pts::rendering
