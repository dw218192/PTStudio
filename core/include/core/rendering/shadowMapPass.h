#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>

namespace pts::rendering {

inline constexpr uint32_t k_max_shadow_maps = 8;
inline constexpr uint32_t k_default_shadow_resolution = 2048;

/// Renders depth maps for shadow-casting lights.
/// Distant lights use an orthographic projection fit to the scene AABB.
/// Rect/disk area lights use a perspective projection from the light's
/// position along its local -Z, with far plane derived from the scene AABB.
/// Sphere and dome lights do not cast shadow maps.
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
    };
    Outputs add_to_frame_graph(FrameGraph& fg, const PassContext& ctx, const Inputs&);

    [[nodiscard]] bool enabled() const {
        return m_enabled;
    }

   private:
    bool m_enabled = true;
    bool m_pcss = true;
    static constexpr uint32_t k_uniform_align = 256;
    uint32_t m_resolution = k_default_shadow_resolution;
};

}  // namespace pts::rendering
