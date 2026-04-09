#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/contactShadowPass.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
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

ContactShadowPass::ContactShadowPass(const ShaderLoader& sl) : IPass(sl) {
}

ContactShadowPass::~ContactShadowPass() {
    release_raw_handles();
}

void ContactShadowPass::release_raw_handles() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bgl) wgpuBindGroupLayoutRelease(ready->bgl);
        if (ready->depth_sampler) wgpuSamplerRelease(ready->depth_sampler);
        if (ready->linear_sampler) wgpuSamplerRelease(ready->linear_sampler);
    }
}

auto ContactShadowPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

static constexpr IPass::DebugTarget k_debug_targets[] = {
    {"Contact Shadow", "contact_shadow"},
};

auto ContactShadowPass::debug_targets() const noexcept -> std::pair<const DebugTarget*, uint32_t> {
    return {k_debug_targets, m_enabled ? 1u : 0u};
}

void ContactShadowPass::do_setup(const webgpu::Device& device) {
    release_raw_handles();

    auto shader_src = get_shader_loader().load("core/generated/shaders/contact_shadow.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    // ── Samplers ──
    WGPUSamplerDescriptor depth_sampler_desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    depth_sampler_desc.magFilter = WGPUFilterMode_Nearest;
    depth_sampler_desc.minFilter = WGPUFilterMode_Nearest;
    depth_sampler_desc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    auto depth_sampler = wgpuDeviceCreateSampler(device.handle(), &depth_sampler_desc);

    WGPUSamplerDescriptor linear_sampler_desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    linear_sampler_desc.magFilter = WGPUFilterMode_Linear;
    linear_sampler_desc.minFilter = WGPUFilterMode_Linear;
    linear_sampler_desc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    auto linear_sampler = wgpuDeviceCreateSampler(device.handle(), &linear_sampler_desc);

    // ── Bind group layout (6 entries) ──
    // 0: uniforms, 1: depth_tex, 2: normals_tex, 3: depth_sampler,
    // 4: linear_sampler, 5: lights
    WGPUBindGroupLayoutEntry entries[6] = {};

    entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    entries[0].buffer.minBindingSize = sizeof(ContactShadowUniforms);

    entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

    entries[2] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Fragment;
    entries[2].texture.sampleType = WGPUTextureSampleType_Float;
    entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

    entries[3] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[3].binding = 3;
    entries[3].visibility = WGPUShaderStage_Fragment;
    entries[3].sampler.type = WGPUSamplerBindingType_NonFiltering;

    entries[4] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[4].binding = 4;
    entries[4].visibility = WGPUShaderStage_Fragment;
    entries[4].sampler.type = WGPUSamplerBindingType_Filtering;

    entries[5] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[5].binding = 5;
    entries[5].visibility = WGPUShaderStage_Fragment;
    entries[5].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[5].buffer.minBindingSize = 0;

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = 6;
    bgl_desc.entries = entries;
    auto bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);

    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &bgl;
    auto pl = wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

    auto pipeline = webgpu::RenderPipelineBuilder(device)
                        .shader(shader)
                        .color_format(WGPUTextureFormat_R8Unorm)
                        .cull_mode(WGPUCullMode_None)
                        .pipeline_layout(pl)
                        .build();
    wgpuPipelineLayoutRelease(pl);

    m_state = Ready{
        std::move(shader), std::move(pipeline), bgl, depth_sampler, linear_sampler,
    };
}

ContactShadowPass::Outputs ContactShadowPass::add_to_frame_graph(FrameGraph& fg,
                                                                 const PassContext& ctx,
                                                                 const Inputs& in) {
    PTS_ZONE_SCOPED;
    if (!m_enabled) return {};
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    // ── Frame graph resources ──
    TextureDesc cs_desc;
    cs_desc.width = ctx.viewport_width;
    cs_desc.height = ctx.viewport_height;
    cs_desc.format = WGPUTextureFormat_R8Unorm;
    cs_desc.clear_color = {1, 1, 1, 1};
    auto cs_handle = create_texture(fg, cs_desc, "contact_shadow");

    BufferDesc uniform_buf_desc;
    uniform_buf_desc.size = sizeof(ContactShadowUniforms);
    uniform_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_handle = create_buffer(fg, uniform_buf_desc, "cs_uniforms");

    // Register bind group (6 entries)
    BindGroupDesc bg_desc;
    bg_desc.layout = ready.bgl;
    bg_desc.entries = {
        {0, ManagedBufferBinding{uniform_buf_handle, 0, sizeof(ContactShadowUniforms)}},
        {1, ManagedTextureBinding{in.depth}},
        {2, ManagedTextureBinding{in.normals}},
        {3, SamplerBinding{ready.depth_sampler}},
        {4, SamplerBinding{ready.linear_sampler}},
        {5, ExternalBufferBinding{in.light_buffer, 0, in.light_buffer_size}},
    };
    auto bg_handle = create_bind_group(fg, std::move(bg_desc), "cs_bg");

    // Capture scalars for lambda
    auto* pipeline = ready.pipeline.handle();
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
        .color(cs_handle)
        .execute([=, &fg](WGPURenderPassEncoder pass) {
            auto uniform_buf = fg.get_buffer_ref(uniform_buf_handle).handle();
            auto bg = fg.get_bind_group_ref(bg_handle).handle();

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

    return {cs_handle};
}

void ContactShadowPass::draw_imgui() {
    ImGui::Checkbox("Enabled", &m_enabled);
    ImGui::SliderFloat("Max Distance", &m_max_distance, 0.01f, 2.0f);
    ImGui::SliderFloat("Thickness", &m_thickness, 0.001f, 0.2f);
    ImGui::SliderFloat("Normal Offset", &m_normal_offset, 0.0f, 0.1f);
    ImGui::SliderInt("Step Count", &m_step_count, 4, 64);
}

}  // namespace pts::rendering
