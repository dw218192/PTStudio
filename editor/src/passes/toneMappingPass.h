#pragma once

#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/texture.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <string_view>
#include <variant>

namespace pts::editor {

class ToneMappingPass final : public rendering::IRenderPass {
   public:
    using IRenderPass::IRenderPass;
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

    void do_setup(const webgpu::Device& device) override;
    void add_to_frame_graph(rendering::FrameGraph& fg, const rendering::PassContext& ctx) override;

    static constexpr uint32_t k_uniform_align = 256;

    // Parameters (controlled from editor UI)
    float m_exposure = 0.0f;  // EV
    uint32_t m_mode = 0;      // 0 = ACES, 1 = Reinhard

   private:
    struct Ready {
        webgpu::ShaderModule shader;
        webgpu::RenderPipeline pipeline;
        webgpu::Buffer uniform_buffer;
        WGPUBindGroupLayout bind_group_layout = nullptr;
        WGPUSampler sampler = nullptr;
        // 1x1 white fallback for when SSAO is unavailable (AO = 1.0)
        webgpu::Texture ssao_fallback_texture;
        WGPUTextureView ssao_fallback_view = nullptr;
        WGPUSampler ssao_sampler = nullptr;
    };

    std::variant<std::monostate, Ready> m_state;
};

}  // namespace pts::editor
