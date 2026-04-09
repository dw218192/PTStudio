#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/outputLayout.h>
#include <core/rendering/passContext.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/toneMappingPass.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
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

ToneMappingPass::~ToneMappingPass() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->descriptor_layout) wgpuBindGroupLayoutRelease(ready->descriptor_layout);
        if (ready->sampler) wgpuSamplerRelease(ready->sampler);
        if (ready->ssao_fallback_view) wgpuTextureViewRelease(ready->ssao_fallback_view);
        if (ready->ssao_sampler) wgpuSamplerRelease(ready->ssao_sampler);
        if (ready->luminance_desc_layout) wgpuBindGroupLayoutRelease(ready->luminance_desc_layout);
        if (ready->depth_fallback_view) wgpuTextureViewRelease(ready->depth_fallback_view);
        if (ready->depth_fallback_tex) wgpuTextureRelease(ready->depth_fallback_tex);
    }
}

auto ToneMappingPass::name() const noexcept -> std::string_view {
    return "tonemapping";
}

auto ToneMappingPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

void ToneMappingPass::do_setup(const webgpu::Device& device) {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->descriptor_layout) wgpuBindGroupLayoutRelease(ready->descriptor_layout);
        if (ready->sampler) wgpuSamplerRelease(ready->sampler);
        if (ready->ssao_fallback_view) wgpuTextureViewRelease(ready->ssao_fallback_view);
        if (ready->ssao_sampler) wgpuSamplerRelease(ready->ssao_sampler);
        if (ready->luminance_desc_layout) wgpuBindGroupLayoutRelease(ready->luminance_desc_layout);
        if (ready->depth_fallback_view) wgpuTextureViewRelease(ready->depth_fallback_view);
        if (ready->depth_fallback_tex) wgpuTextureRelease(ready->depth_fallback_tex);
    }

    // --- Tone mapping render pipeline ---
    auto shader_src = get_shader_loader().load("editor/generated/shaders/tonemapping.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    // Bind group layout: uniform + hdr texture + hdr sampler + ssao texture + ssao sampler +
    // exposure result
    auto tone_internal =
        create_output_layout(device, {
                                         OutputSlot::uniform(sizeof(ToneMappingUniforms)),
                                         OutputSlot::texture(WGPUTextureFormat_RGBA16Float),
                                         OutputSlot::sampler(WGPUSamplerBindingType_Filtering),
                                         OutputSlot::texture(WGPUTextureFormat_RGBA8Unorm),
                                         OutputSlot::sampler(WGPUSamplerBindingType_Filtering),
                                         OutputSlot::storage(sizeof(ExposureResult)),
                                     });
    auto descriptor_layout = tone_internal.layout;
    tone_internal.layout = nullptr;
    tone_internal.release();

    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &descriptor_layout;
    auto pipeline_layout = wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

    auto pipeline = webgpu::RenderPipelineBuilder(device)
                        .shader(shader)
                        .color_format(WGPUTextureFormat_RGBA8Unorm)
                        .cull_mode(WGPUCullMode_None)
                        .pipeline_layout(pipeline_layout)
                        .build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    // HDR linear sampler
    WGPUSamplerDescriptor sampler_desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    sampler_desc.magFilter = WGPUFilterMode_Linear;
    sampler_desc.minFilter = WGPUFilterMode_Linear;
    sampler_desc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    auto sampler = wgpuDeviceCreateSampler(device.handle(), &sampler_desc);

    // SSAO sampler (clamp-to-edge)
    WGPUSamplerDescriptor ssao_sampler_desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    ssao_sampler_desc.magFilter = WGPUFilterMode_Linear;
    ssao_sampler_desc.minFilter = WGPUFilterMode_Linear;
    ssao_sampler_desc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    ssao_sampler_desc.addressModeU = WGPUAddressMode_ClampToEdge;
    ssao_sampler_desc.addressModeV = WGPUAddressMode_ClampToEdge;
    auto ssao_sampler = wgpuDeviceCreateSampler(device.handle(), &ssao_sampler_desc);

    // 1x1 white R8Unorm fallback (AO = 1.0 everywhere when SSAO is off)
    WGPUTextureDescriptor fb_tex_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    fb_tex_desc.size = {1, 1, 1};
    fb_tex_desc.format = WGPUTextureFormat_R8Unorm;
    fb_tex_desc.usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst);
    fb_tex_desc.mipLevelCount = 1;
    fb_tex_desc.sampleCount = 1;
    fb_tex_desc.dimension = WGPUTextureDimension_2D;
    auto fb_raw = wgpuDeviceCreateTexture(device.handle(), &fb_tex_desc);
    INVARIANT_MSG(fb_raw, "Failed to create SSAO fallback texture");

    uint8_t white = 255;
    WGPUTexelCopyBufferLayout fb_layout = {};
    fb_layout.bytesPerRow = 1;
    fb_layout.rowsPerImage = 1;
    WGPUTexelCopyTextureInfo fb_dest = {};
    fb_dest.texture = fb_raw;
    fb_dest.aspect = WGPUTextureAspect_All;
    WGPUExtent3D fb_extent = {1, 1, 1};
    wgpuQueueWriteTexture(device.queue(), &fb_dest, &white, sizeof(white), &fb_layout, &fb_extent);

    WGPUTextureViewDescriptor fb_view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    fb_view_desc.format = WGPUTextureFormat_R8Unorm;
    fb_view_desc.dimension = WGPUTextureViewDimension_2D;
    fb_view_desc.mipLevelCount = 1;
    fb_view_desc.arrayLayerCount = 1;
    auto fb_view = wgpuTextureCreateView(fb_raw, &fb_view_desc);
    INVARIANT_MSG(fb_view, "Failed to create SSAO fallback texture view");

    // --- Luminance compute pipeline ---
    auto lum_shader_src = get_shader_loader().load("editor/generated/shaders/luminance.wgsl");
    auto luminance_shader = device.create_shader_module_from_source(lum_shader_src);

    auto lum_internal = create_output_layout(
        device,
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
    auto luminance_desc_layout = lum_internal.layout;
    lum_internal.layout = nullptr;
    lum_internal.release();

    WGPUPipelineLayoutDescriptor lum_pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    lum_pl_desc.bindGroupLayoutCount = 1;
    lum_pl_desc.bindGroupLayouts = &luminance_desc_layout;
    auto lum_pipeline_layout = wgpuDeviceCreatePipelineLayout(device.handle(), &lum_pl_desc);

    auto luminance_pipeline = webgpu::ComputePipelineBuilder(device)
                                  .shader(luminance_shader)
                                  .entry_point("cs_main")
                                  .pipeline_layout(lum_pipeline_layout)
                                  .build();

    wgpuPipelineLayoutRelease(lum_pipeline_layout);

    // 1x1 Depth32Float fallback (value 0.0 = not sky) for when scene_depth unavailable
    WGPUTextureDescriptor df_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    df_desc.size = {1, 1, 1};
    df_desc.format = WGPUTextureFormat_Depth32Float;
    df_desc.usage = WGPUTextureUsage_TextureBinding;
    df_desc.mipLevelCount = 1;
    df_desc.dimension = WGPUTextureDimension_2D;
    auto depth_fallback_tex = wgpuDeviceCreateTexture(device.handle(), &df_desc);

    WGPUTextureViewDescriptor df_view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    df_view_desc.format = WGPUTextureFormat_Depth32Float;
    df_view_desc.dimension = WGPUTextureViewDimension_2D;
    df_view_desc.mipLevelCount = 1;
    df_view_desc.arrayLayerCount = 1;
    auto depth_fallback_view = wgpuTextureCreateView(depth_fallback_tex, &df_view_desc);

    m_state = Ready{
        std::move(shader),
        std::move(pipeline),
        descriptor_layout,
        sampler,
        webgpu::Texture(fb_raw),
        fb_view,
        ssao_sampler,
        std::move(luminance_shader),
        std::move(luminance_pipeline),
        luminance_desc_layout,
        depth_fallback_tex,
        depth_fallback_view,
    };
}

void ToneMappingPass::add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    // Compute delta time for temporal smoothing
    float dt = ctx.time - m_prev_time;
    m_prev_time = ctx.time;
    if (dt <= 0.0f || dt > 1.0f) dt = 1.0f / 60.0f;

    // Detect auto-exposure toggle: reset result buffer on re-enable
    bool needs_reset = m_auto_exposure && !m_prev_auto_exposure;
    m_prev_auto_exposure = m_auto_exposure;

    // Read HDR input, write LDR tone_mapped_color
    PRECONDITION(m_inputs.hdr_color.is_valid());
    auto hdr_handle = m_inputs.hdr_color;

    TextureDesc ldr_desc;
    ldr_desc.width = ctx.viewport_width;
    ldr_desc.height = ctx.viewport_height;
    ldr_desc.format = WGPUTextureFormat_RGBA8Unorm;
    ldr_desc.clear_color = {0, 0, 0, 1};
    ldr_desc.usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc);
    auto ldr_handle = create_texture(fg, ldr_desc, "ldr_output");
    m_ldr_output = ldr_handle;

    // SSAO from pass inputs (threaded by renderer, not global lookup)
    auto ssao_found = m_inputs.ssao;

    // Exposure result buffer (persistent across frames, pass-scoped key)
    BufferDesc result_buf_desc{};
    result_buf_desc.size = sizeof(ExposureResult);
    result_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    auto result_buf_handle = create_buffer(fg, result_buf_desc, "auto_exposure_result");

    // --- Luminance compute pass (only when auto-exposure is on) ---
    if (m_auto_exposure) {
        BufferDesc lum_params_desc;
        lum_params_desc.size = sizeof(LuminanceParams);
        lum_params_desc.usage =
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
        auto lum_params_handle = create_buffer(fg, lum_params_desc, "lum_params");

        // Depth for sky masking (optional — path tracer may not have it)
        auto depth_handle = m_inputs.depth;
        bool has_depth = depth_handle.has_value();

        auto lum_bg_bld = descriptor(fg, ready.luminance_desc_layout, "lum_bg")
                              .texture(0, hdr_handle)
                              .sampler(1, ready.sampler)
                              .buffer(2, result_buf_handle, 0, sizeof(ExposureResult))
                              .buffer(3, lum_params_handle, 0, sizeof(LuminanceParams));
        if (has_depth) {
            lum_bg_bld.texture(4, *depth_handle);
        } else {
            lum_bg_bld.external_view(4, ready.depth_fallback_view);
        }
        auto lum_bg_handle = lum_bg_bld.build();

        auto* lum_pipeline = ready.luminance_pipeline.handle();
        auto queue = ctx.queue;
        auto width = ctx.viewport_width;
        auto height = ctx.viewport_height;
        auto adaptation_speed = m_adaptation_speed;

        auto lum_builder = fg.add_pass("luminance");
        lum_builder.read(hdr_handle);
        if (has_depth) {
            lum_builder.read(*depth_handle);
        }

        lum_builder.execute([=, &fg](WGPUComputePassEncoder enc) {
            auto result_buf = fg.get_buffer_ref(result_buf_handle).handle();
            auto lum_params_buf = fg.get_buffer_ref(lum_params_handle).handle();
            auto lum_bg = fg.get_descriptor_ref(lum_bg_handle).handle();

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
    auto uniform_buf_handle = create_buffer(fg, buf_desc, "uniforms");

    // Register descriptor (6 entries)
    auto bg_builder = descriptor(fg, ready.descriptor_layout, "bg0")
                          .buffer(0, uniform_buf_handle, 0, sizeof(ToneMappingUniforms))
                          .texture(1, hdr_handle)
                          .sampler(2, ready.sampler);
    if (ssao_found) {
        bg_builder.texture(3, *ssao_found);
    } else {
        bg_builder.external_view(3, ready.ssao_fallback_view);
    }
    auto bg_handle = bg_builder.sampler(4, ready.ssao_sampler)
                         .buffer(5, result_buf_handle, 0, sizeof(ExposureResult))
                         .build();

    auto* pipeline_handle = ready.pipeline.handle();
    auto queue = ctx.queue;
    auto exposure = m_exposure;
    auto mode = m_mode;
    auto auto_exposure_enabled = m_auto_exposure;

    auto builder = fg.add_pass("tonemapping");
    builder.read(hdr_handle);
    builder.color(ldr_handle);
    if (ssao_found) {
        builder.read(*ssao_found);
    }

    builder.execute([=, &fg](WGPURenderPassEncoder pass) {
        auto uniform_buf = fg.get_buffer_ref(uniform_buf_handle).handle();
        auto desc_group = fg.get_descriptor_ref(bg_handle).handle();

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
