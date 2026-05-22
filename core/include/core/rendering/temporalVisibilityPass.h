#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>

namespace pts::rendering {

class FallbackPool;
class TemporalStorageManager;

/// Per-pixel temporal accumulation of PCSS shadow visibility for the first
/// shadow-casting light. Reads scene depth + the live shadow map, computes
/// PCSS visibility for that one light, and EMA-blends with a persistent
/// history texture vended by TemporalStorageManager.
///
/// POC scope (see ticket temporal-shadow-visibility-poc):
///   * Single shadow-casting light (other lights keep inline PCSS in forward).
///   * No motion vectors / disocclusion -- assumes stationary camera.
///   * Output is a single R16Float texture; downstream consumers sample it
///     in place of compute_shadow_pcss for that one light.
class TemporalVisibilityPass final : public IPass {
   public:
    using IPass::IPass;

    TemporalVisibilityPass(const TemporalVisibilityPass&) = delete;
    TemporalVisibilityPass& operator=(const TemporalVisibilityPass&) = delete;
    TemporalVisibilityPass(TemporalVisibilityPass&&) = delete;
    TemporalVisibilityPass& operator=(TemporalVisibilityPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "temporal_visibility";
    }

    void draw_imgui() override;

    struct Inputs {
        TextureDeclHandle depth;
        TextureDeclHandle shadow_array;
        BufferDeclHandle shadow_info;
        /// Index into shadow_infos for the light whose visibility we accumulate.
        /// UINT32_MAX means "no shadow-casting light" -> pass returns invalid.
        uint32_t shadow_light_index = UINT32_MAX;
    };
    struct Outputs {
        /// Accumulated visibility (R16Float, persistent ping-pong). Invalid
        /// when the pass is disabled or when no shadow-casting light exists;
        /// the caller is responsible for binding a fallback view in that case.
        TextureDeclHandle accumulated_visibility;
    };

    Outputs add_to_frame_graph(FrameGraph& fg, const PassContext& ctx, const Inputs& in,
                               TemporalStorageManager& storage);

    [[nodiscard]] bool enabled() const {
        return m_enabled;
    }

    /// Reset the cold-start bootstrap counter. Call after a viewport resize or
    /// camera teleport so the next frame uses curr_vis directly (alpha=1)
    /// instead of blending into the stale history.
    void reset_history() {
        m_frame_counter = 0;
    }

    bool m_enabled = true;
    /// EMA weight on the new sample once warmed up. The first frame after
    /// reset uses alpha=1 to bootstrap; subsequent frames use this value.
    float m_blend_weight = 0.05f;

   private:
    uint64_t m_frame_counter = 0;
    uint32_t m_history_width = 0;
    uint32_t m_history_height = 0;
};

}  // namespace pts::rendering
