#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/fallbackPool.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/outputLayout.h>
#include <core/rendering/passContext.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/toneMappingPass.h>
#include <core/rendering/webgpu/device.h>
#include <imgui.h>

using namespace pts;
using namespace pts::rendering;

struct ToneMappingUniforms {
    float exposure;
    uint32_t mode;
    uint32_t auto_exposure_enabled;
    uint32_t _pad1;
};
static_assert(sizeof(ToneMappingUniforms) == 16);

struct LuminanceParams {
    uint32_t width;
    uint32_t height;
    float adaptation_speed;
    float dt;
    uint32_t has_depth;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
};
static_assert(sizeof(LuminanceParams) == 32);

struct ExposureResult {
    float auto_exposure;
    uint32_t frame_count;
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(ExposureResult) == 16);

auto ToneMappingPass::name() const noexcept -> std::string_view {
    return "tonemapping";
}

void ToneMappingPass::add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) {
    PTS_ZONE_SCOPED;
    ensure_initialized(ctx.device);

    // --- Tone mapping render pipeline ---
    auto descriptor_layout = fg.bind_group_layout(
        "tonemapping/desc", {
                                OutputSlot::uniform(sizeof(ToneMappingUniforms)),
                                OutputSlot::texture(WGPUTextureFormat_RGBA16Float),
                                OutputSlot::sampler(WGPUSamplerBindingType_Filtering),
                                OutputSlot::texture(WGPUTextureFormat_RGBA8Unorm),
                                OutputSlot::sampler(WGPUSamplerBindingType_Filtering),
                                OutputSlot::storage(sizeof(ExposureResult)),
                            });

    auto* pipeline_handle = fg.render_pipeline("tonemapping")
                                .shader("editor/generated/shaders/tonemapping.wgsl")
                                .color_format(WGPUTextureFormat_RGBA8Unorm)
                                .cull_mode(WGPUCullMode_None)
                                .bind_group_layouts({descriptor_layout})
                                .build();

    // --- Luminance compute pipeline ---
    auto luminance_desc_layout = fg.bind_group_layout(
        "tonemapping/luminance",
        {
            OutputSlot::texture(WGPUTextureFormat_RGBA16Float).visibility(WGPUShaderStage_Compute),
            OutputSlot::sampler(WGPUSamplerBindingType_Filtering)
                .visibility(WGPUShaderStage_Compute),
            OutputSlot::storage(sizeof(ExposureResult))
                .read_write()
                .visibility(WGPUShaderStage_Compute),
            OutputSlot::uniform(sizeof(LuminanceParams)).visibility(WGPUShaderStage_Compute),
            OutputSlot::texture(WGPUTextureFormat_Depth32Float).visibility(WGPUShaderStage_Compute),
        });

    auto* lum_pipeline = fg.compute_pipeline("luminance")
                             .shader("editor/generated/shaders/luminance.wgsl")
                             .entry_point("cs_main")
                             .bind_group_layouts({luminance_desc_layout})
                             .build();

    // Compute delta time for temporal smoothing
    float dt = ctx.time - m_prev_time;
    m_prev_time = ctx.time;
    if (dt <= 0.0f || dt > 1.0f) dt = 1.0f / 60.0f;

    // Detect auto-exposure toggle: reset result buffer on re-enable
    bool needs_reset = m_auto_exposure && !m_prev_auto_exposure;
    m_prev_auto_exposure = m_auto_exposure;

    // Read HDR input, write LDR tone_mapped_color
    PRECONDITION(m_inputs.hdr_color);
    auto hdr_decl = m_inputs.hdr_color;

    TextureDesc ldr_desc;
    ldr_desc.width = ctx.viewport_width;
    ldr_desc.height = ctx.viewport_height;
    ldr_desc.format = WGPUTextureFormat_RGBA8Unorm;
    ldr_desc.clear_color = {0, 0, 0, 1};
    ldr_desc.usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc);
    auto ldr_decl = create_texture(fg, ldr_desc, "ldr_output");
    m_ldr_output = ldr_decl;

    // SSAO from pass inputs (threaded by renderer, not global lookup)
    auto ssao_decl = m_inputs.ssao;

    // Exposure result buffer (persistent across frames, pass-scoped key)
    BufferDesc result_buf_desc{};
    result_buf_desc.size = sizeof(ExposureResult);
    result_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    auto result_buf_decl = create_buffer(fg, result_buf_desc, "auto_exposure_result");

    // --- Luminance compute pass (only when auto-exposure is on) ---
    if (m_auto_exposure) {
        BufferDesc lum_params_desc;
        lum_params_desc.size = sizeof(LuminanceParams);
        lum_params_desc.usage =
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
        auto lum_params_decl = create_buffer(fg, lum_params_desc, "lum_params");

        // Depth for sky masking (optional — path tracer may not have it)
        auto depth_decl = m_inputs.depth;
        bool has_depth = static_cast<bool>(depth_decl);

        auto lum_bg_bld = descriptor(fg, luminance_desc_layout, "lum_bg")
                              .texture(0, hdr_decl)
                              .sampler(1, fg.sampler(WGPUSamplerBindingType_Filtering))
                              .buffer(2, result_buf_decl, 0, sizeof(ExposureResult))
                              .buffer(3, lum_params_decl, 0, sizeof(LuminanceParams));
        if (has_depth) {
            lum_bg_bld.texture(4, depth_decl);
        } else {
            lum_bg_bld.external_view(4, fg.fallback_pool().view(WGPUTextureFormat_Depth32Float,
                                                                WGPUTextureViewDimension_2D));
        }
        auto lum_bg_decl = lum_bg_bld.build();

        auto queue = ctx.queue;
        auto width = ctx.viewport_width;
        auto height = ctx.viewport_height;
        auto adaptation_speed = m_adaptation_speed;

        auto lum_builder = fg.add_pass("luminance");
        lum_builder.read(hdr_decl);
        if (has_depth) {
            lum_builder.read(depth_decl);
        }

        lum_builder.execute([=](rendering::ExecuteContext& exec, WGPUComputePassEncoder enc) {
            auto result_buf = exec.get(result_buf_decl).buffer;
            auto lum_params_buf = exec.get(lum_params_decl).buffer;
            auto lum_bg = exec.get(lum_bg_decl).bind_group;

            // Reset result buffer when auto-exposure was just re-enabled
            if (needs_reset) {
                ExposureResult zeros{};
                wgpuQueueWriteBuffer(queue, result_buf, 0, &zeros, sizeof(zeros));
            }

            LuminanceParams params{};
            params.width = width;
            params.height = height;
            params.adaptation_speed = adaptation_speed;
            params.dt = dt;
            params.has_depth = has_depth ? 1u : 0u;
            wgpuQueueWriteBuffer(queue, lum_params_buf, 0, &params, sizeof(params));

            wgpuComputePassEncoderSetPipeline(enc, lum_pipeline);
            wgpuComputePassEncoderSetBindGroup(enc, 0, lum_bg, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(enc, 1, 1, 1);
        });
    }

    // --- Tone mapping render pass ---

    // Register uniform buffer
    BufferDesc buf_desc;
    buf_desc.size = sizeof(ToneMappingUniforms);
    buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_decl = create_buffer(fg, buf_desc, "uniforms");

    // Register descriptor (6 entries)
    auto bg_builder = descriptor(fg, descriptor_layout, "bg0")
                          .buffer(0, uniform_buf_decl, 0, sizeof(ToneMappingUniforms))
                          .texture(1, hdr_decl)
                          .sampler(2, fg.sampler(WGPUSamplerBindingType_Filtering));
    if (ssao_decl) {
        bg_builder.texture(3, ssao_decl);
    } else {
        bg_builder.external_view(
            3, fg.fallback_pool().view(WGPUTextureFormat_RGBA8Unorm, WGPUTextureViewDimension_2D));
    }
    auto bg_decl = bg_builder.sampler(4, fg.sampler(WGPUSamplerBindingType_Filtering))
                       .buffer(5, result_buf_decl, 0, sizeof(ExposureResult))
                       .build();

    auto queue = ctx.queue;
    auto exposure = m_exposure;
    auto mode = m_mode;
    auto auto_exposure_enabled = m_auto_exposure;

    auto builder = fg.add_pass("tonemapping");
    builder.read(hdr_decl);
    builder.color(ldr_decl);
    if (ssao_decl) {
        builder.read(ssao_decl);
    }

    builder.execute([=](rendering::ExecuteContext& exec, WGPURenderPassEncoder pass) {
        auto uniform_buf = exec.get(uniform_buf_decl).buffer;
        auto desc_group = exec.get(bg_decl).bind_group;

        ToneMappingUniforms uniforms{};
        uniforms.exposure = exposure;
        uniforms.mode = mode;
        uniforms.auto_exposure_enabled = auto_exposure_enabled ? 1u : 0u;
        wgpuQueueWriteBuffer(queue, uniform_buf, 0, &uniforms, sizeof(uniforms));

        wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, desc_group, 0, nullptr);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    });
}

void ToneMappingPass::draw_imgui() {
    int mode_int = static_cast<int>(m_mode);
    const char* mode_names[] = {"ACES Filmic", "Reinhard"};
    if (ImGui::Combo("Mode", &mode_int, mode_names, 2)) {
        m_mode = static_cast<uint32_t>(mode_int);
    }
    ImGui::Checkbox("Auto Exposure", &m_auto_exposure);
    if (m_auto_exposure) {
        ImGui::SliderFloat("Adaptation Speed", &m_adaptation_speed, 0.1f, 10.0f);
    }
}
