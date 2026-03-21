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
    }

    auto shader_src = get_shader_loader().load("editor/generated/shaders/tonemapping.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    auto uniform_buffer = device.create_buffer(
        sizeof(ToneMappingUniforms),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    // Bind group layout: uniform buffer + texture + sampler
    WGPUBindGroupLayoutEntry entries[3] = {};

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

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = 3;
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

    // Create linear sampler
    WGPUSamplerDescriptor sampler_desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    sampler_desc.magFilter = WGPUFilterMode_Linear;
    sampler_desc.minFilter = WGPUFilterMode_Linear;
    sampler_desc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    auto sampler = wgpuDeviceCreateSampler(device.handle(), &sampler_desc);

    m_state = Ready{
        std::move(shader), std::move(pipeline), std::move(uniform_buffer), bind_group_layout,
        sampler,
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

    // Upload uniforms
    ToneMappingUniforms uniforms{};
    uniforms.exposure = m_exposure;
    uniforms.mode = m_mode;
    wgpuQueueWriteBuffer(ctx.queue, ready.uniform_buffer.handle(), 0, &uniforms, sizeof(uniforms));

    auto* pipeline_handle = ready.pipeline.handle();
    auto uniform_buf = ready.uniform_buffer.handle();
    auto bgl = ready.bind_group_layout;
    auto sampler = ready.sampler;
    auto dev = ctx.device.handle();

    fg.add_pass("tonemapping")
        .read(hdr_handle)
        .color(ldr_handle)
        .execute([=, &fg](WGPURenderPassEncoder pass) {
            auto hdr_ref = fg.get_texture_ref(hdr_handle);
            if (!hdr_ref) return;

            WGPUBindGroupEntry bg_entries[3] = {};
            bg_entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
            bg_entries[0].binding = 0;
            bg_entries[0].buffer = uniform_buf;
            bg_entries[0].size = sizeof(ToneMappingUniforms);

            bg_entries[1] = WGPU_BIND_GROUP_ENTRY_INIT;
            bg_entries[1].binding = 1;
            bg_entries[1].textureView = hdr_ref.view();

            bg_entries[2] = WGPU_BIND_GROUP_ENTRY_INIT;
            bg_entries[2].binding = 2;
            bg_entries[2].sampler = sampler;

            WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
            bg_desc.layout = bgl;
            bg_desc.entryCount = 3;
            bg_desc.entries = bg_entries;
            auto bind_group = wgpuDeviceCreateBindGroup(dev, &bg_desc);

            wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);  // fullscreen triangle

            wgpuBindGroupRelease(bind_group);
        });
}
