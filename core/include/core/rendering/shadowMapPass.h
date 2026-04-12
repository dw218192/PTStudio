#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/outputLayout.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <vector>

namespace pts::rendering {

inline constexpr uint32_t k_max_shadow_maps = 4;
inline constexpr uint32_t k_default_shadow_resolution = 2048;

/// Renders depth maps for shadow-casting distant lights.
class ShadowMapPass final : public IPass {
   public:
    using IPass::IPass;

    ShadowMapPass(const ShadowMapPass&) = delete;
    ShadowMapPass& operator=(const ShadowMapPass&) = delete;
    ShadowMapPass(ShadowMapPass&&) = delete;
    ShadowMapPass& operator=(ShadowMapPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "shadow_map";
    }
    [[nodiscard]] auto requires_viewport() const noexcept -> bool override {
        return false;
    }

    void draw_imgui() override;

    struct Inputs {};
    struct Outputs {
        TextureDeclHandle shadow_array;
        BufferDeclHandle shadow_info;
        DescriptorDeclHandle consumer_desc;
    };
    Outputs add_to_frame_graph(FrameGraph& fg, const PassContext& ctx, const Inputs&);

    /// Slot declarations for the consumer bind group (shadow receiver).
    /// Renderers pass these to FrameGraph::bind_group_layout() to obtain
    /// the BGL for pipeline layout creation.
    [[nodiscard]] static std::vector<OutputSlot> consumer_slots();

    [[nodiscard]] bool enabled() const {
        return m_enabled;
    }

   private:
    bool m_enabled = true;
    static constexpr uint32_t k_uniform_align = 256;
    uint32_t m_resolution = k_default_shadow_resolution;
};

}  // namespace pts::rendering
