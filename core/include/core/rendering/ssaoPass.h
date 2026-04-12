#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>

namespace pts::rendering {

class FallbackPool;
class ShaderLoader;

/// Screen-space ambient occlusion pass.
/// Reads scene_depth (Depth32Float) and scene_normals (RG16Float),
/// writes ssao (R8Unorm) via two sub-passes: AO generation
/// and bilateral blur.
class SSAOPass final : public IPass {
   public:
    using IPass::IPass;

    SSAOPass(const SSAOPass&) = delete;
    SSAOPass& operator=(const SSAOPass&) = delete;
    SSAOPass(SSAOPass&&) = delete;
    SSAOPass& operator=(SSAOPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "ssao";
    }
    [[nodiscard]] auto debug_targets() const noexcept
        -> std::pair<const DebugTarget*, uint32_t> override;

    struct Inputs {
        TextureDeclHandle depth;
        TextureDeclHandle normals;
    };
    struct Outputs {
        TextureDeclHandle ssao;
    };

    Outputs add_to_frame_graph(FrameGraph& fg, const PassContext& ctx, const Inputs& in,
                               FallbackPool& fallbacks);
    void draw_imgui() override;

    // Tunable parameters (exposed via ImGui)
    bool m_enabled = true;
    float m_radius = 0.5f;
    float m_bias = 0.025f;
    float m_intensity = 1.0f;
    int m_sample_count = 32;

   private:
    static constexpr uint32_t k_max_kernel_size = 64;
};

}  // namespace pts::rendering
