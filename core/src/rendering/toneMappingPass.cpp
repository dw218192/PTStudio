#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
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
        if (ready->bind_group_layout) wgpuBindGroupLayoutRelease(ready->bind_group_layout);
        if (ready->sampler) wgpuSamplerRelease(ready->sampler);
        if (ready->ssao_fallback_view) wgpuTextureViewRelease(ready->ssao_fallback_view);
        if (ready->ssao_sampler) wgpuSamplerRelease(ready->ssao_sampler);
        if (ready->luminance_bgl) wgpuBindGroupLayoutRelease(ready->luminance_bgl);
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
        if (ready->bind_group_layout) wgpuBindGroupLayoutRelease(ready->bind_group_layout);
        if (ready->sampler) wgpuSamplerRelease(ready->sampler);
        if (ready->ssao_fallback_view) wgpuTextureViewRelease(ready->ssao_fallback_view);
        if (ready->ssao_sampler) wgpuSamplerRelease(ready->ssao_sampler);
        if (ready->luminance_bgl) wgpuBindGroupLayoutRelease(ready->luminance_bgl);
        if (ready->depth_fallback_view) wgpuTextureViewRelease(ready->depth_fallback_view);
        if (ready->depth_fallback_tex) wgpuTextureRelease(ready->depth_fallback_tex);
    }

    // --- Tone mapping render pipeline ---
    auto shader_src = get_shader_loader().load("editor/generated/shaders/tonemapping.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    // Bind group layout: uniform + hdr texture + hdr sampler + ssao texture + ssao sampler +
    // exposure result
    WGPUBindGroupLayoutEntry entries[6] = {};

    entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    entries[0].buffer.minBindingSize = sizeof(ToneMappingUniforms);

    entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].texture.sampleType = WGPUTextureSampleType_Float;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

    entries[2] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Fragment;
    entries[2].sampler.type = WGPUSamplerBindingType_Filtering;

    entries[3] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[3].binding = 3;
    entries[3].visibility = WGPUShaderStage_Fragment;
    entries[3].texture.sampleType = WGPUTextureSampleType_Float;
    entries[3].texture.viewDimension = WGPUTextureViewDimension_2D;

    entries[4] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[4].binding = 4;
    entries[4].visibility = WGPUShaderStage_Fragment;
    entries[4].sampler.type = WGPUSamplerBindingType_Filtering;

    entries[5] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[5].binding = 5;
    entries[5].visibility = WGPUShaderStage_Fragment;
    entries[5].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[5].buffer.minBindingSize = sizeof(ExposureResult);

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = 6;
    bgl_desc.entries = entries;
    auto bind_group_layout = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);

    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &bind_group_layout;
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

    WGPUBindGroupLayoutEntry lum_entries[5] = {};

    lum_entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    lum_entries[0].binding = 0;
    lum_entries[0].visibility = WGPUShaderStage_Compute;
    lum_entries[0].texture.sampleType = WGPUTextureSampleType_Float;
    lum_entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;

    lum_entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    lum_entries[1].binding = 1;
    lum_entries[1].visibility = WGPUShaderStage_Compute;
    lum_entries[1].sampler.type = WGPUSamplerBindingType_Filtering;

    lum_entries[2] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    lum_entries[2].binding = 2;
    lum_entries[2].visibility = WGPUShaderStage_Compute;
    lum_entries[2].buffer.type = WGPUBufferBindingType_Storage;
    lum_entries[2].buffer.minBindingSize = sizeof(ExposureResult);

    lum_entries[3] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    lum_entries[3].binding = 3;
    lum_entries[3].visibility = WGPUShaderStage_Compute;
    lum_entries[3].buffer.type = WGPUBufferBindingType_Uniform;
    lum_entries[3].buffer.minBindingSize = sizeof(LuminanceParams);

    lum_entries[4] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    lum_entries[4].binding = 4;
    lum_entries[4].visibility = WGPUShaderStage_Compute;
    lum_entries[4].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
    lum_entries[4].texture.viewDimension = WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor lum_bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    lum_bgl_desc.entryCount = 5;
    lum_bgl_desc.entries = lum_entries;
    auto luminance_bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &lum_bgl_desc);

    WGPUPipelineLayoutDescriptor lum_pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    lum_pl_desc.bindGroupLayoutCount = 1;
    lum_pl_desc.bindGroupLayouts = &luminance_bgl;
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
        std::move(shader), std::move(pipeline),         bind_group_layout,
        sampler,           webgpu::Texture(fb_raw),     fb_view,
        ssao_sampler,      std::move(luminance_shader), std::move(luminance_pipeline),
        luminance_bgl,     depth_fallback_tex,          depth_fallback_view,
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

    // Check if SSAOPass produced the "ssao" resource this frame
    auto ssao_found = fg.find("ssao");

    // Exposure result buffer (persistent across frames)
    BufferDesc result_buf_desc;
    result_buf_desc.size = sizeof(ExposureResult);
    result_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    auto result_buf_handle = fg.find_or_create_buffer("auto_exposure_result", result_buf_desc);

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

        BindGroupDesc lum_bg_desc;
        lum_bg_desc.layout = ready.luminance_bgl;
        lum_bg_desc.entries.resize(5);
        lum_bg_desc.entries[0].binding = 0;
        lum_bg_desc.entries[0].texture = hdr_handle;
        lum_bg_desc.entries[1].binding = 1;
        lum_bg_desc.entries[1].sampler = ready.sampler;
        lum_bg_desc.entries[2].binding = 2;
        lum_bg_desc.entries[2].buffer = result_buf_handle;
        lum_bg_desc.entries[2].buffer_size = sizeof(ExposureResult);
        lum_bg_desc.entries[3].binding = 3;
        lum_bg_desc.entries[3].buffer = lum_params_handle;
        lum_bg_desc.entries[3].buffer_size = sizeof(LuminanceParams);
        lum_bg_desc.entries[4].binding = 4;
        if (has_depth) {
            lum_bg_desc.entries[4].texture = *depth_handle;
        } else {
            lum_bg_desc.entries[4].external_view = ready.depth_fallback_view;
        }
        auto lum_bg_handle = create_bind_group(fg, std::move(lum_bg_desc), "lum_bg");

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
            auto lum_bg = fg.get_bind_group_ref(lum_bg_handle).handle();

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

    // Register bind group (6 entries)
    BindGroupDesc bg_desc;
    bg_desc.layout = ready.bind_group_layout;
    bg_desc.entries.resize(6);
    bg_desc.entries[0].binding = 0;
    bg_desc.entries[0].buffer = uniform_buf_handle;
    bg_desc.entries[0].buffer_size = sizeof(ToneMappingUniforms);
    bg_desc.entries[1].binding = 1;
    bg_desc.entries[1].texture = hdr_handle;
    bg_desc.entries[2].binding = 2;
    bg_desc.entries[2].sampler = ready.sampler;
    if (ssao_found) {
        bg_desc.entries[3].binding = 3;
        bg_desc.entries[3].texture = *ssao_found;
    } else {
        bg_desc.entries[3].binding = 3;
        bg_desc.entries[3].external_view = ready.ssao_fallback_view;
    }
    bg_desc.entries[4].binding = 4;
    bg_desc.entries[4].sampler = ready.ssao_sampler;
    bg_desc.entries[5].binding = 5;
    bg_desc.entries[5].buffer = result_buf_handle;
    bg_desc.entries[5].buffer_size = sizeof(ExposureResult);
    auto bg_handle = create_bind_group(fg, std::move(bg_desc), "bg0");

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
        auto bind_group = fg.get_bind_group_ref(bg_handle).handle();

        ToneMappingUniforms uniforms{};
        uniforms.exposure = exposure;
        uniforms.mode = mode;
        uniforms.auto_exposure_enabled = auto_exposure_enabled ? 1u : 0u;
        wgpuQueueWriteBuffer(queue, uniform_buf, 0, &uniforms, sizeof(uniforms));

        wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);
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
