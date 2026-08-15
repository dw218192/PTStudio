#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/bilateralBlur.h>
#include <core/rendering/fallbackPool.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/gbufferPass.h>
#include <core/rendering/passContext.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/ssaoPass.h>
#include <core/rendering/webgpu/device.h>
#include <imgui.h>
#include <ssao_shader_metadata.h>

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
    uint32_t _pad[2];          // 152: 8  -> total 160
};
static_assert(sizeof(SSAOUniforms) == 160, "SSAOUniforms must match shader std140 layout");

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

static constexpr IPass::DebugTarget k_debug_targets[] = {
    {"AO", "ssao"},
};

auto SSAOPass::debug_targets() const noexcept -> std::pair<const DebugTarget*, uint32_t> {
    return {k_debug_targets, m_enabled ? 1u : 0u};
}

SSAOPass::Outputs SSAOPass::add_to_frame_graph(FrameGraph& fg, const PassContext& ctx,
                                               const Inputs& in, FallbackPool& fallbacks) {
    PTS_ZONE_SCOPED;
    if (!m_enabled) return {};
    ensure_initialized(ctx.device);

    // -- Kernel buffer (persistent -- first-call upload) --
    // The initial data must outlive the first compile(); store it in a static
    // buffer that persists for the process lifetime.
    static const auto k_kernel_data = [] {
        std::array<glm::vec4, k_max_kernel_size> k{};
        generate_kernel(k.data(), k_max_kernel_size);
        return k;
    }();
    {
        BufferDesc desc;
        desc.size = sizeof(k_kernel_data);
        desc.usage =
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
        fg.buffer("ssao_kernel", desc, k_kernel_data.data());
    }

    // -- Noise texture (4x4 RGBA8Unorm, persistent) --
    static const auto k_noise_data = [] {
        std::array<uint8_t, 4 * 4 * 4> d{};
        generate_noise_data(d.data());
        return d;
    }();
    {
        WGPUTextureDescriptor noise_tex_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        noise_tex_desc.size = {4, 4, 1};
        noise_tex_desc.format = WGPUTextureFormat_RGBA8Unorm;
        noise_tex_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding |
                                                             WGPUTextureUsage_CopyDst);
        noise_tex_desc.mipLevelCount = 1;
        noise_tex_desc.sampleCount = 1;
        noise_tex_desc.dimension = WGPUTextureDimension_2D;
        fg.texture("ssao_noise", noise_tex_desc, k_noise_data.data(), k_noise_data.size(), 4 * 4);
    }

    auto gen_descl = fg.bind_group_layout(
        "ssao/gen", ssao_shader::create_bind_group_layout_0(ctx.device.handle()));

    auto* gen_pipeline = fg.render_pipeline("ssao_gen")
                             .shader("core/generated/shaders/ssao.wgsl")
                             .color_format(WGPUTextureFormat_R8Unorm)
                             .cull_mode(WGPUCullMode_None)
                             .bind_group_layouts({gen_descl})
                             .build();

    // -- Frame graph resources --
    TextureDesc r8_desc;
    r8_desc.width = ctx.viewport_width;
    r8_desc.height = ctx.viewport_height;
    r8_desc.format = WGPUTextureFormat_R8Unorm;
    r8_desc.clear_color = {1, 1, 1, 1};

    auto depth_decl = in.depth;
    auto normals_decl = in.normals;
    auto ssao_raw_decl = create_texture(fg, r8_desc, "ssao_raw");

    BufferDesc gen_buf_desc;
    gen_buf_desc.size = sizeof(SSAOUniforms);
    gen_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto gen_uniform_buf_decl = create_buffer(fg, gen_buf_desc, "gen_uniforms");

    // Look up persistent resources (bumps their last_declared_frame)
    auto kernel_decl = fg.find_buffer("ssao_kernel");
    auto noise_decl = fg.find_texture("ssao_noise");
    INVARIANT(kernel_decl && noise_decl);

    // AO gen descriptor via DescriptorBuilder
    auto gen_desc_decl =
        descriptor(fg, gen_descl, "gen_desc")
            .texture(0, depth_decl)
            .sampler(1, fg.sampler(WGPUSamplerBindingType_NonFiltering))
            .texture(2, normals_decl)
            .sampler(3, fg.sampler(WGPUSamplerBindingType_Filtering))
            .buffer(4, gen_uniform_buf_decl, 0, sizeof(SSAOUniforms))
            .texture(5, noise_decl)
            .sampler(6, fg.sampler(WGPUSamplerBindingType_NonFiltering, WGPUAddressMode_Repeat))
            .buffer(7, kernel_decl)
            .build();

    // Capture scalars for lambdas
    auto queue = ctx.queue;
    auto proj_matrix = ctx.proj_matrix;
    auto viewport_width = ctx.viewport_width;
    auto viewport_height = ctx.viewport_height;
    auto radius = m_radius;
    auto bias = m_bias;
    auto intensity = m_intensity;
    auto sample_count = m_sample_count;

    // -- Pass 1: AO Generation --
    fg.add_pass("ssao_gen")
        .read(depth_decl)
        .read(normals_decl)
        .color(ssao_raw_decl)
        .execute([=](ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto gen_uniform_buf = exec.get(gen_uniform_buf_decl).buffer;
            auto gen_desc = exec.get(gen_desc_decl).bind_group;

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
            wgpuRenderPassEncoderSetBindGroup(pass, 0, gen_desc, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

    // -- Pass 2: Bilateral Blur --
    BilateralBlurParams blur_params;
    blur_params.input = ssao_raw_decl;
    blur_params.depth = depth_decl;
    blur_params.output_format = WGPUTextureFormat_RGBA8Unorm;
    blur_params.depth_threshold = 0.001f;
    blur_params.debug_label = "ssao/blur";
    auto ssao_decl = add_bilateral_blur(fg, ctx, blur_params);

    return {ssao_decl};
}

void SSAOPass::draw_imgui() {
    ImGui::Checkbox("Enabled", &m_enabled);
    ImGui::SliderFloat("Radius", &m_radius, 0.01f, 2.0f);
    ImGui::SliderFloat("Bias", &m_bias, 0.0f, 0.1f);
    ImGui::SliderFloat("Intensity", &m_intensity, 0.0f, 5.0f);
}

}  // namespace pts::rendering
