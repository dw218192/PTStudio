#pragma once

#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <string_view>

namespace pts::rendering {

class ToneMappingPass final : public IPass {
   public:
    using IPass::IPass;

    ToneMappingPass(const ToneMappingPass&) = delete;
    ToneMappingPass& operator=(const ToneMappingPass&) = delete;
    ToneMappingPass(ToneMappingPass&&) = delete;
    ToneMappingPass& operator=(ToneMappingPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto requires_viewport() const noexcept -> bool override {
        return true;
    }

    struct Inputs {
        TextureDeclHandle hdr_color;
        TextureDeclHandle depth;  // optional; for auto-exposure sky masking
        TextureDeclHandle ssao;   // optional; ambient occlusion (from SSAOPass)
    };
    void set_inputs(const Inputs& in) {
        m_inputs = in;
    }

    /// LDR tone-mapped output. Valid after add_to_frame_graph.
    [[nodiscard]] TextureDeclHandle ldr_output() const {
        return m_ldr_output;
    }

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
    TextureDeclHandle m_ldr_output;
    bool m_prev_auto_exposure = false;
    float m_prev_time = 0.0f;
};

}  // namespace pts::rendering
