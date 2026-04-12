#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/outputLayout.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/webgpu.h>

#include <array>
#include <cstdint>

namespace pts::rendering {

class FallbackPool;
class ShaderLoader;

/// Screen-space contact shadow pass.
/// Reads scene_depth (Depth32Float), scene_normals (RG16Float), and the light
/// buffer, writes contact_shadow (R8Unorm, 1=lit, 0=shadowed) by ray-marching
/// the depth buffer toward each non-dome light.
class ContactShadowPass final : public IPass {
   public:
    using IPass::IPass;

    ContactShadowPass(const ContactShadowPass&) = delete;
    ContactShadowPass& operator=(const ContactShadowPass&) = delete;
    ContactShadowPass(ContactShadowPass&&) = delete;
    ContactShadowPass& operator=(ContactShadowPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "contact_shadow";
    }
    [[nodiscard]] auto debug_targets() const noexcept
        -> std::pair<const DebugTarget*, uint32_t> override;

    struct Inputs {
        TextureDeclHandle depth;
        TextureDeclHandle normals;
        WGPUBuffer light_buffer = nullptr;
        uint64_t light_buffer_size = 0;
    };
    struct Outputs {
        TextureDeclHandle contact_shadow;
        DescriptorDeclHandle consumer_desc;
    };

    Outputs add_to_frame_graph(FrameGraph& fg, const PassContext& ctx, const Inputs& in,
                               FallbackPool& fallbacks);
    void draw_imgui() override;

    /// Slot declarations for the consumer bind group (CS texture + sampler).
    [[nodiscard]] static std::array<OutputSlot, 2> consumer_slots();

    // Tunable parameters (exposed via ImGui)
    bool m_enabled = true;
    float m_max_distance = 0.5f;
    float m_thickness = 0.05f;
    float m_normal_offset = 0.01f;
    int m_step_count = 16;
};

}  // namespace pts::rendering
