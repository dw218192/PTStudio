#pragma once

#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/texture.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <variant>

namespace pts::rendering {

class ToneMappingPass final : public IPass {
   public:
    using IPass::IPass;
    ~ToneMappingPass() override;

    ToneMappingPass(const ToneMappingPass&) = delete;
    ToneMappingPass& operator=(const ToneMappingPass&) = delete;
    ToneMappingPass(ToneMappingPass&&) = delete;
    ToneMappingPass& operator=(ToneMappingPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto is_ready() const noexcept -> bool override;
    [[nodiscard]] auto requires_viewport() const noexcept -> bool override {
        return true;
    }

    struct Inputs {
        TextureHandle hdr_color;
        std::optional<TextureHandle> depth;  // for auto-exposure sky masking
        std::optional<TextureHandle> ssao;   // ambient occlusion (from SSAOPass)
    };
    void set_inputs(const Inputs& in) {
        m_inputs = in;
    }

    /// LDR tone-mapped output. Valid after add_to_frame_graph.
    [[nodiscard]] TextureHandle ldr_output() const {
        return m_ldr_output;
    }

    void do_setup(const webgpu::Device& device) override;
    void add_to_frame_graph(FrameGraph& fg, const PassContext& ctx);
    void draw_imgui() override;

    static constexpr uint32_t k_uniform_align = 256;

    // Parameters (controlled from renderer UI)
    float m_exposure = 0.0f;  // EV bias (additive on top of auto-exposure when enabled)
    uint32_t m_mode = 0;      // 0 = ACES, 1 = Reinhard
    bool m_auto_exposure = true;
    float m_adaptation_speed = 2.0f;

   private:
    Inputs m_inputs;
    TextureHandle m_ldr_output;

    struct Ready {
        // Tone mapping render pipeline
        webgpu::ShaderModule shader;
        webgpu::RenderPipeline pipeline;
        WGPUBindGroupLayout bind_group_layout = nullptr;
        WGPUSampler sampler = nullptr;
        // 1x1 white fallback for when SSAO is unavailable (AO = 1.0)
        webgpu::Texture ssao_fallback_texture;
        WGPUTextureView ssao_fallback_view = nullptr;
        WGPUSampler ssao_sampler = nullptr;

        // Luminance compute pipeline
        webgpu::ShaderModule luminance_shader;
        webgpu::ComputePipeline luminance_pipeline;
        WGPUBindGroupLayout luminance_bgl = nullptr;
        // 1x1 depth fallback (value 0.0 = not sky) for when scene_depth unavailable
        WGPUTexture depth_fallback_tex = nullptr;
        WGPUTextureView depth_fallback_view = nullptr;
    };

    std::variant<std::monostate, Ready> m_state;
    bool m_prev_auto_exposure = false;
    float m_prev_time = 0.0f;
};

}  // namespace pts::rendering
