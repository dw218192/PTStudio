#include "toneMappingPass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/webgpu/pipelineBuilder.h>

using namespace pts;
using namespace pts::editor;

struct ToneMappingUniforms {
    float exposure;
    uint32_t mode;
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(ToneMappingUniforms) == 16);

ToneMappingPass::~ToneMappingPass() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group_layout) wgpuBindGroupLayoutRelease(ready->bind_group_layout);
        if (ready->sampler) wgpuSamplerRelease(ready->sampler);
        if (ready->ssao_fallback_view) wgpuTextureViewRelease(ready->ssao_fallback_view);
        if (ready->ssao_sampler) wgpuSamplerRelease(ready->ssao_sampler);
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
    }

    auto shader_src = get_shader_loader().load("editor/generated/shaders/tonemapping.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    // Bind group layout: uniform + hdr texture + hdr sampler + ssao texture + ssao sampler
    WGPUBindGroupLayoutEntry entries[5] = {};

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

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = 5;
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

    m_state = Ready{
        std::move(shader), std::move(pipeline), bind_group_layout, sampler, webgpu::Texture(fb_raw),
        fb_view,           ssao_sampler,
    };
}

void ToneMappingPass::add_to_frame_graph(rendering::FrameGraph& fg,
                                         const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    // Read HDR scene_color, write LDR tone_mapped_color
    rendering::TextureDesc hdr_desc;
    hdr_desc.width = ctx.viewport_width;
    hdr_desc.height = ctx.viewport_height;
    hdr_desc.format = WGPUTextureFormat_RGBA16Float;
    auto hdr_handle = fg.find_or_create("scene_color", hdr_desc);

    rendering::TextureDesc ldr_desc;
    ldr_desc.width = ctx.viewport_width;
    ldr_desc.height = ctx.viewport_height;
    ldr_desc.format = WGPUTextureFormat_RGBA8Unorm;
    ldr_desc.clear_color = {0, 0, 0, 1};
    auto ldr_handle = fg.find_or_create("tone_mapped_color", ldr_desc);

    // Check if SSAOPass produced the "ssao" resource this frame
    auto ssao_found = fg.find("ssao");

    // Register uniform buffer
    rendering::BufferDesc buf_desc;
    buf_desc.size = sizeof(ToneMappingUniforms);
    buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_handle = create_buffer(fg, buf_desc, "uniforms");

    // Register bind group (5 entries)
    rendering::BindGroupDesc bg_desc;
    bg_desc.layout = ready.bind_group_layout;
    bg_desc.entries.resize(5);
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
    auto bg_handle = create_bind_group(fg, std::move(bg_desc), "bg0");

    auto* pipeline_handle = ready.pipeline.handle();
    auto queue = ctx.queue;
    auto exposure = m_exposure;
    auto mode = m_mode;

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
        wgpuQueueWriteBuffer(queue, uniform_buf, 0, &uniforms, sizeof(uniforms));

        wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    });
}
