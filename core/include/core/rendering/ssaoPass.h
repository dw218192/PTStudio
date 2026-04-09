#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/texture.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <variant>

namespace pts::rendering {

class ShaderLoader;

/// Screen-space ambient occlusion pass.
/// Reads scene_depth (Depth32Float) and scene_normals (RG16Float),
/// writes ssao (R8Unorm) via two sub-passes: AO generation
/// and bilateral blur.
class SSAOPass final : public IPass {
   public:
    explicit SSAOPass(const ShaderLoader& sl);
    ~SSAOPass() override;

    SSAOPass(const SSAOPass&) = delete;
    SSAOPass& operator=(const SSAOPass&) = delete;
    SSAOPass(SSAOPass&&) = delete;
    SSAOPass& operator=(SSAOPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "ssao";
    }
    [[nodiscard]] auto is_ready() const noexcept -> bool override;
    [[nodiscard]] auto debug_targets() const noexcept
        -> std::pair<const DebugTarget*, uint32_t> override;

    struct Inputs {
        ResourceHandle depth;
        ResourceHandle normals;
    };
    struct Outputs {
        ResourceHandle ssao;
    };

    void do_setup(const webgpu::Device& device) override;
    Outputs add_to_frame_graph(FrameGraph& fg, const PassContext& ctx, const Inputs& in);
    void draw_imgui() override;

    // Tunable parameters (exposed via ImGui)
    bool m_enabled = true;
    float m_radius = 0.5f;
    float m_bias = 0.025f;
    float m_intensity = 1.0f;
    int m_sample_count = 32;

   private:
    static constexpr uint32_t k_max_kernel_size = 64;

    struct Ready {
        // AO generation
        webgpu::ShaderModule gen_shader;
        webgpu::RenderPipeline gen_pipeline;
        WGPUBindGroupLayout gen_desc_layout = nullptr;

        // Blur
        webgpu::ShaderModule blur_shader;
        webgpu::RenderPipeline blur_pipeline;
        WGPUBindGroupLayout blur_desc_layout = nullptr;

        // Noise texture (4x4 RGBA8Unorm)
        webgpu::Texture noise_texture;
        WGPUTextureView noise_view = nullptr;

        // Samplers
        WGPUSampler depth_sampler = nullptr;   // non-filtering
        WGPUSampler linear_sampler = nullptr;  // linear filtering
        WGPUSampler noise_sampler = nullptr;   // repeat wrapping

        // Sample kernel (hemisphere vectors)
        webgpu::Buffer kernel_buffer;
    };

    void release_raw_handles();

    std::variant<std::monostate, Ready> m_state;
};

}  // namespace pts::rendering
