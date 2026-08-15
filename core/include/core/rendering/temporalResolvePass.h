#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>

namespace pts::rendering {

class TemporalStorageManager;

/// Temporal resolve pass (the "resolve" half of the gen/resolve shadow
/// pipeline). Reads the raw per-frame visibility texture from
/// ShadowVisibilityPass plus the previous frame's resolved history,
/// reprojects the history with the G-buffer motion vector, applies a 3x3
/// neighborhood variance clamp to the history sample, then EMA-blends it
/// with the raw sample. The result ping-pongs through TemporalStorageManager.
///
/// Motion vector convention: motion = prev_uv - curr_uv (see gbuffer.slang).
/// Pixels whose reprojected UV falls outside [0,1] bypass the history blend
/// and emit only the raw sample (alpha = 1 path).
class TemporalResolvePass final : public IPass {
   public:
    using IPass::IPass;

    TemporalResolvePass(const TemporalResolvePass&) = delete;
    TemporalResolvePass& operator=(const TemporalResolvePass&) = delete;
    TemporalResolvePass(TemporalResolvePass&&) = delete;
    TemporalResolvePass& operator=(TemporalResolvePass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "temporal_resolve";
    }

    void draw_imgui() override;

    struct Inputs {
        /// Raw per-frame visibility from ShadowVisibilityPass.
        TextureDeclHandle raw_visibility;
        /// Screen-space motion vectors (RG16Float, prev_uv - curr_uv) from
        /// GBufferPass. When invalid the pass falls back to same-UV history
        /// sampling -- useful for tests that don't run the full pre-pass set.
        TextureDeclHandle motion;
    };
    struct Outputs {
        /// Resolved visibility (R16Float). When the pass is disabled this is
        /// the raw input passed through unchanged; invalid when the raw input
        /// is invalid.
        TextureDeclHandle resolved_visibility;
    };

    Outputs add_to_frame_graph(FrameGraph& fg, const PassContext& ctx, const Inputs& in,
                               TemporalStorageManager& storage);

    [[nodiscard]] bool enabled() const {
        return m_enabled;
    }

    /// Reset the cold-start bootstrap counter. Call after a viewport resize or
    /// camera teleport so the next frame uses the raw sample directly
    /// (alpha=1) instead of blending into stale history.
    void reset_history() {
        m_frame_counter = 0;
    }

    bool m_enabled = true;
    /// EMA weight on the new sample once warmed up. The first frame after
    /// reset uses alpha=1 to bootstrap; subsequent frames use this value.
    float m_blend_weight = 0.05f;
    /// Variance-clamp width in standard deviations: history is clamped to
    /// neighborhood mean +/- gamma*stddev. Smaller = tighter clamp (less
    /// ghosting, more flicker); larger = looser (more ghosting).
    float m_gamma = 1.0f;

   private:
    uint64_t m_frame_counter = 0;
    uint32_t m_history_width = 0;
    uint32_t m_history_height = 0;
};

}  // namespace pts::rendering
