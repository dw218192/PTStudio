#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/ssaoPass.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <imgui.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <random>

namespace pts::rendering {

// Must match SSAOUniforms in ssao.slang (std140 layout).
struct SSAOUniforms {
    glm::mat4 projection;      // 0:  64
    glm::mat4 inv_projection;  // 64: 64
    glm::vec2 viewport_size;   // 128: 8
    float radius;              // 136: 4
    float bias;                // 140: 4
    float intensity;           // 144: 4
    int32_t sample_count;      // 148: 4
    uint32_t _pad[2];          // 152: 8  → total 160
};
static_assert(sizeof(SSAOUniforms) == 160, "SSAOUniforms must match shader std140 layout");

// Must match BlurUniforms in ssao_blur.slang.
struct SSAOBlurUniforms {
    glm::vec2 texel_size;  // 0: 8
    float _pad[2];         // 8: 8  → total 16
};
static_assert(sizeof(SSAOBlurUniforms) == 16, "SSAOBlurUniforms must match shader std140 layout");

namespace {

void generate_kernel(glm::vec4* out, uint32_t count) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (uint32_t i = 0; i < count; ++i) {
        glm::vec3 s(dist(rng) * 2.0f - 1.0f,  // x: [-1, 1]
                    dist(rng) * 2.0f - 1.0f,  // y: [-1, 1]
                    dist(rng));               // z: [0, 1] hemisphere

        s = glm::normalize(s);
        s *= dist(rng);

        // Importance sampling: more samples near the surface
        float scale = float(i) / float(count);
        scale = glm::mix(0.1f, 1.0f, scale * scale);
        s *= scale;

        out[i] = glm::vec4(s, 0.0f);
    }
}

void generate_noise_data(uint8_t* out) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * 3.14159265358979323846f);

    for (int i = 0; i < 16; ++i) {
        float angle = angle_dist(rng);
        float x = std::cos(angle);
        float y = std::sin(angle);
        out[i * 4 + 0] = static_cast<uint8_t>((x * 0.5f + 0.5f) * 255.0f);
        out[i * 4 + 1] = static_cast<uint8_t>((y * 0.5f + 0.5f) * 255.0f);
        out[i * 4 + 2] = 0;
        out[i * 4 + 3] = 255;
    }
}

}  // namespace

SSAOPass::SSAOPass(const ShaderLoader& sl) : IPass(sl) {
}

SSAOPass::~SSAOPass() {
    release_raw_handles();
}

void SSAOPass::release_raw_handles() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->gen_bgl) wgpuBindGroupLayoutRelease(ready->gen_bgl);
        if (ready->blur_bgl) wgpuBindGroupLayoutRelease(ready->blur_bgl);
        if (ready->noise_view) wgpuTextureViewRelease(ready->noise_view);
        if (ready->depth_sampler) wgpuSamplerRelease(ready->depth_sampler);
        if (ready->linear_sampler) wgpuSamplerRelease(ready->linear_sampler);
        if (ready->noise_sampler) wgpuSamplerRelease(ready->noise_sampler);
    }
}

auto SSAOPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

static constexpr IPass::DebugTarget k_debug_targets[] = {
    {"AO", "ssao"},
};

auto SSAOPass::debug_targets() const noexcept -> std::pair<const DebugTarget*, uint32_t> {
    return {k_debug_targets, m_enabled ? 1u : 0u};
}

void SSAOPass::do_setup(const webgpu::Device& device) {
    release_raw_handles();

    auto gen_src = get_shader_loader().load("core/generated/shaders/ssao.wgsl");
    auto gen_shader = device.create_shader_module_from_source(gen_src);

    auto blur_src = get_shader_loader().load("core/generated/shaders/ssao_blur.wgsl");
    auto blur_shader = device.create_shader_module_from_source(blur_src);

    // ── Kernel buffer ──
    std::array<glm::vec4, k_max_kernel_size> kernel_data{};
    generate_kernel(kernel_data.data(), k_max_kernel_size);

    auto kernel_buffer = device.create_buffer(
        sizeof(glm::vec4) * k_max_kernel_size,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
    wgpuQueueWriteBuffer(device.queue(), kernel_buffer.handle(), 0, kernel_data.data(),
                         sizeof(glm::vec4) * k_max_kernel_size);

    // ── Noise texture (4×4 RGBA8Unorm) ──
    uint8_t noise_data[4 * 4 * 4];
    generate_noise_data(noise_data);

    WGPUTextureDescriptor noise_tex_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    noise_tex_desc.size = {4, 4, 1};
    noise_tex_desc.format = WGPUTextureFormat_RGBA8Unorm;
    noise_tex_desc.usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst);
    noise_tex_desc.mipLevelCount = 1;
    noise_tex_desc.sampleCount = 1;
    noise_tex_desc.dimension = WGPUTextureDimension_2D;
    auto noise_raw = wgpuDeviceCreateTexture(device.handle(), &noise_tex_desc);
    INVARIANT_MSG(noise_raw, "Failed to create SSAO noise texture");

    WGPUTexelCopyBufferLayout layout = {};
    layout.bytesPerRow = 4 * 4;  // 4 pixels × 4 bytes
    layout.rowsPerImage = 4;
    WGPUTexelCopyTextureInfo dest = {};
    dest.texture = noise_raw;
    dest.aspect = WGPUTextureAspect_All;
    WGPUExtent3D extent = {4, 4, 1};
    wgpuQueueWriteTexture(device.queue(), &dest, noise_data, sizeof(noise_data), &layout, &extent);

    WGPUTextureViewDescriptor noise_view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    noise_view_desc.format = WGPUTextureFormat_RGBA8Unorm;
    noise_view_desc.dimension = WGPUTextureViewDimension_2D;
    noise_view_desc.mipLevelCount = 1;
    noise_view_desc.arrayLayerCount = 1;
    auto noise_view = wgpuTextureCreateView(noise_raw, &noise_view_desc);
    INVARIANT_MSG(noise_view, "Failed to create SSAO noise texture view");

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

    WGPUSamplerDescriptor noise_sampler_desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    noise_sampler_desc.magFilter = WGPUFilterMode_Nearest;
    noise_sampler_desc.minFilter = WGPUFilterMode_Nearest;
    noise_sampler_desc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    noise_sampler_desc.addressModeU = WGPUAddressMode_Repeat;
    noise_sampler_desc.addressModeV = WGPUAddressMode_Repeat;
    auto noise_sampler = wgpuDeviceCreateSampler(device.handle(), &noise_sampler_desc);

    // ── AO Generation BGL ──
    // 0: uniforms, 1: depth, 2: normals, 3: noise, 4: depth_sampler,
    // 5: linear_sampler, 6: noise_sampler, 7: kernel
    WGPUBindGroupLayoutEntry gen_entries[8] = {};

    gen_entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    gen_entries[0].binding = 0;
    gen_entries[0].visibility = WGPUShaderStage_Fragment;
    gen_entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    gen_entries[0].buffer.minBindingSize = sizeof(SSAOUniforms);

    gen_entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    gen_entries[1].binding = 1;
    gen_entries[1].visibility = WGPUShaderStage_Fragment;
    gen_entries[1].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
    gen_entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

    gen_entries[2] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    gen_entries[2].binding = 2;
    gen_entries[2].visibility = WGPUShaderStage_Fragment;
    gen_entries[2].texture.sampleType = WGPUTextureSampleType_Float;
    gen_entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

    gen_entries[3] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    gen_entries[3].binding = 3;
    gen_entries[3].visibility = WGPUShaderStage_Fragment;
    gen_entries[3].texture.sampleType = WGPUTextureSampleType_Float;
    gen_entries[3].texture.viewDimension = WGPUTextureViewDimension_2D;

    gen_entries[4] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    gen_entries[4].binding = 4;
    gen_entries[4].visibility = WGPUShaderStage_Fragment;
    gen_entries[4].sampler.type = WGPUSamplerBindingType_NonFiltering;

    gen_entries[5] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    gen_entries[5].binding = 5;
    gen_entries[5].visibility = WGPUShaderStage_Fragment;
    gen_entries[5].sampler.type = WGPUSamplerBindingType_Filtering;

    gen_entries[6] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    gen_entries[6].binding = 6;
    gen_entries[6].visibility = WGPUShaderStage_Fragment;
    gen_entries[6].sampler.type = WGPUSamplerBindingType_NonFiltering;

    gen_entries[7] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    gen_entries[7].binding = 7;
    gen_entries[7].visibility = WGPUShaderStage_Fragment;
    gen_entries[7].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    gen_entries[7].buffer.minBindingSize = sizeof(glm::vec4) * k_max_kernel_size;

    WGPUBindGroupLayoutDescriptor gen_bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    gen_bgl_desc.entryCount = 8;
    gen_bgl_desc.entries = gen_entries;
    auto gen_bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &gen_bgl_desc);

    WGPUPipelineLayoutDescriptor gen_pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    gen_pl_desc.bindGroupLayoutCount = 1;
    gen_pl_desc.bindGroupLayouts = &gen_bgl;
    auto gen_pl = wgpuDeviceCreatePipelineLayout(device.handle(), &gen_pl_desc);

    auto gen_pipeline = webgpu::RenderPipelineBuilder(device)
                            .shader(gen_shader)
                            .color_format(WGPUTextureFormat_R8Unorm)
                            .cull_mode(WGPUCullMode_None)
                            .pipeline_layout(gen_pl)
                            .build();
    wgpuPipelineLayoutRelease(gen_pl);

    // ── Blur BGL ──
    // 0: uniforms, 1: ssao_raw, 2: depth, 3: linear_sampler, 4: depth_sampler
    WGPUBindGroupLayoutEntry blur_entries[5] = {};

    blur_entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    blur_entries[0].binding = 0;
    blur_entries[0].visibility = WGPUShaderStage_Fragment;
    blur_entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    blur_entries[0].buffer.minBindingSize = sizeof(SSAOBlurUniforms);

    blur_entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    blur_entries[1].binding = 1;
    blur_entries[1].visibility = WGPUShaderStage_Fragment;
    blur_entries[1].texture.sampleType = WGPUTextureSampleType_Float;
    blur_entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

    blur_entries[2] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    blur_entries[2].binding = 2;
    blur_entries[2].visibility = WGPUShaderStage_Fragment;
    blur_entries[2].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
    blur_entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

    blur_entries[3] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    blur_entries[3].binding = 3;
    blur_entries[3].visibility = WGPUShaderStage_Fragment;
    blur_entries[3].sampler.type = WGPUSamplerBindingType_Filtering;

    blur_entries[4] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    blur_entries[4].binding = 4;
    blur_entries[4].visibility = WGPUShaderStage_Fragment;
    blur_entries[4].sampler.type = WGPUSamplerBindingType_NonFiltering;

    WGPUBindGroupLayoutDescriptor blur_bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    blur_bgl_desc.entryCount = 5;
    blur_bgl_desc.entries = blur_entries;
    auto blur_bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &blur_bgl_desc);

    WGPUPipelineLayoutDescriptor blur_pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    blur_pl_desc.bindGroupLayoutCount = 1;
    blur_pl_desc.bindGroupLayouts = &blur_bgl;
    auto blur_pl = wgpuDeviceCreatePipelineLayout(device.handle(), &blur_pl_desc);

    auto blur_pipeline = webgpu::RenderPipelineBuilder(device)
                             .shader(blur_shader)
                             .color_format(WGPUTextureFormat_RGBA8Unorm)
                             .cull_mode(WGPUCullMode_None)
                             .pipeline_layout(blur_pl)
                             .build();
    wgpuPipelineLayoutRelease(blur_pl);

    m_state = Ready{
        std::move(gen_shader),
        std::move(gen_pipeline),
        gen_bgl,
        std::move(blur_shader),
        std::move(blur_pipeline),
        blur_bgl,
        webgpu::Texture(noise_raw),
        noise_view,
        depth_sampler,
        linear_sampler,
        noise_sampler,
        std::move(kernel_buffer),
    };
}

SSAOPass::Outputs SSAOPass::add_to_frame_graph(FrameGraph& fg, const PassContext& ctx,
                                               const Inputs& in) {
    PTS_ZONE_SCOPED;
    if (!m_enabled) return {};
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    // ── Frame graph resources ──
    TextureDesc r8_desc;
    r8_desc.width = ctx.viewport_width;
    r8_desc.height = ctx.viewport_height;
    r8_desc.format = WGPUTextureFormat_R8Unorm;
    r8_desc.clear_color = {1, 1, 1, 1};

    auto depth_handle = in.depth;
    auto normals_handle = in.normals;
    auto ssao_raw_handle = create_texture(fg, r8_desc, "ssao_raw");

    TextureDesc ao_desc = r8_desc;
    ao_desc.format = WGPUTextureFormat_RGBA8Unorm;
    auto ssao_handle = fg.find_or_create("ssao", ao_desc);

    // Register uniform buffers with frame graph
    BufferDesc gen_buf_desc;
    gen_buf_desc.size = sizeof(SSAOUniforms);
    gen_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto gen_uniform_buf_handle = create_buffer(fg, gen_buf_desc, "gen_uniforms");

    BufferDesc blur_buf_desc;
    blur_buf_desc.size = sizeof(SSAOBlurUniforms);
    blur_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto blur_uniform_buf_handle = create_buffer(fg, blur_buf_desc, "blur_uniforms");

    // Register AO gen bind group (8 entries)
    auto kernel_buf = ready.kernel_buffer.handle();
    BindGroupDesc gen_bg_desc;
    gen_bg_desc.layout = ready.gen_bgl;
    gen_bg_desc.entries.resize(8);
    gen_bg_desc.entries[0].binding = 0;
    gen_bg_desc.entries[0].buffer = gen_uniform_buf_handle;
    gen_bg_desc.entries[0].buffer_size = sizeof(SSAOUniforms);
    gen_bg_desc.entries[1].binding = 1;
    gen_bg_desc.entries[1].texture = depth_handle;
    gen_bg_desc.entries[2].binding = 2;
    gen_bg_desc.entries[2].texture = normals_handle;
    gen_bg_desc.entries[3].binding = 3;
    gen_bg_desc.entries[3].external_view = ready.noise_view;
    gen_bg_desc.entries[4].binding = 4;
    gen_bg_desc.entries[4].sampler = ready.depth_sampler;
    gen_bg_desc.entries[5].binding = 5;
    gen_bg_desc.entries[5].sampler = ready.linear_sampler;
    gen_bg_desc.entries[6].binding = 6;
    gen_bg_desc.entries[6].sampler = ready.noise_sampler;
    gen_bg_desc.entries[7].binding = 7;
    gen_bg_desc.entries[7].external_buffer = kernel_buf;
    gen_bg_desc.entries[7].external_buffer_size = sizeof(glm::vec4) * k_max_kernel_size;
    auto gen_bg_handle = create_bind_group(fg, std::move(gen_bg_desc), "gen_bg");

    // Register blur bind group (5 entries)
    BindGroupDesc blur_bg_desc;
    blur_bg_desc.layout = ready.blur_bgl;
    blur_bg_desc.entries.resize(5);
    blur_bg_desc.entries[0].binding = 0;
    blur_bg_desc.entries[0].buffer = blur_uniform_buf_handle;
    blur_bg_desc.entries[0].buffer_size = sizeof(SSAOBlurUniforms);
    blur_bg_desc.entries[1].binding = 1;
    blur_bg_desc.entries[1].texture = ssao_raw_handle;
    blur_bg_desc.entries[2].binding = 2;
    blur_bg_desc.entries[2].texture = depth_handle;
    blur_bg_desc.entries[3].binding = 3;
    blur_bg_desc.entries[3].sampler = ready.linear_sampler;
    blur_bg_desc.entries[4].binding = 4;
    blur_bg_desc.entries[4].sampler = ready.depth_sampler;
    auto blur_bg_handle = create_bind_group(fg, std::move(blur_bg_desc), "blur_bg");

    // Capture scalars for lambdas
    auto* gen_pipeline = ready.gen_pipeline.handle();
    auto* blur_pipeline = ready.blur_pipeline.handle();
    auto queue = ctx.queue;
    auto proj_matrix = ctx.proj_matrix;
    auto viewport_width = ctx.viewport_width;
    auto viewport_height = ctx.viewport_height;
    auto radius = m_radius;
    auto bias = m_bias;
    auto intensity = m_intensity;
    auto sample_count = m_sample_count;

    // ── Pass 1: AO Generation ──
    fg.add_pass("ssao_gen")
        .read(depth_handle)
        .read(normals_handle)
        .color(ssao_raw_handle)
        .execute([=, &fg](WGPURenderPassEncoder pass) {
            auto gen_uniform_buf = fg.get_buffer_ref(gen_uniform_buf_handle).handle();
            auto gen_bg = fg.get_bind_group_ref(gen_bg_handle).handle();

            SSAOUniforms uniforms{};
            uniforms.projection = proj_matrix;
            uniforms.inv_projection = glm::inverse(proj_matrix);
            uniforms.viewport_size = {
                static_cast<float>(viewport_width),
                static_cast<float>(viewport_height),
            };
            uniforms.radius = radius;
            uniforms.bias = bias;
            uniforms.intensity = intensity;
            uniforms.sample_count = sample_count;
            wgpuQueueWriteBuffer(queue, gen_uniform_buf, 0, &uniforms, sizeof(uniforms));

            wgpuRenderPassEncoderSetPipeline(pass, gen_pipeline);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, gen_bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

    // ── Pass 2: Bilateral Blur ──
    fg.add_pass("ssao_blur")
        .read(ssao_raw_handle)
        .read(depth_handle)
        .color(ssao_handle)
        .execute([=, &fg](WGPURenderPassEncoder pass) {
            auto blur_uniform_buf = fg.get_buffer_ref(blur_uniform_buf_handle).handle();
            auto blur_bg = fg.get_bind_group_ref(blur_bg_handle).handle();

            SSAOBlurUniforms blur_u{};
            blur_u.texel_size = {1.0f / static_cast<float>(viewport_width),
                                 1.0f / static_cast<float>(viewport_height)};
            wgpuQueueWriteBuffer(queue, blur_uniform_buf, 0, &blur_u, sizeof(blur_u));

            wgpuRenderPassEncoderSetPipeline(pass, blur_pipeline);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, blur_bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

    return {ssao_handle};
}

void SSAOPass::draw_imgui() {
    ImGui::Checkbox("Enabled", &m_enabled);
    ImGui::SliderFloat("Radius", &m_radius, 0.01f, 2.0f);
    ImGui::SliderFloat("Bias", &m_bias, 0.0f, 0.1f);
    ImGui::SliderFloat("Intensity", &m_intensity, 0.0f, 5.0f);
}

}  // namespace pts::rendering
