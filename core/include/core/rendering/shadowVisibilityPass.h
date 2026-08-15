#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>

namespace pts::rendering {

/// Shadow-visibility generation pass (the "gen" half of the gen/resolve
/// shadow pipeline). Reconstructs world position from the scene depth buffer,
/// frustum-guards, and runs PCSS for the single shadow-casting light into a
/// raw R16Float visibility texture. No temporal blend -- this is the
/// technique-agnostic seam: a future RT/SDF shadow technique replaces only
/// this pass, leaving TemporalResolvePass downstream unchanged.
///
/// Single-light scope: secondary shadow-casting lights are not handled here
/// (multi-light shadows are deferred to clustered lighting).
class ShadowVisibilityPass final : public IPass {
   public:
    using IPass::IPass;

    ShadowVisibilityPass(const ShadowVisibilityPass&) = delete;
    ShadowVisibilityPass& operator=(const ShadowVisibilityPass&) = delete;
    ShadowVisibilityPass(ShadowVisibilityPass&&) = delete;
    ShadowVisibilityPass& operator=(ShadowVisibilityPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "shadow_visibility";
    }

    void draw_imgui() override;

    struct Inputs {
        TextureDeclHandle depth;
        TextureDeclHandle shadow_array;
        BufferDeclHandle shadow_info;
        /// Index into shadow_infos for the light whose visibility we generate.
        /// UINT32_MAX means "no shadow-casting light" -> pass returns invalid.
        uint32_t shadow_light_index = UINT32_MAX;
    };
    struct Outputs {
        /// Raw per-frame visibility (R16Float, transient). Invalid when the
        /// pass is disabled or no shadow-casting light exists; the caller is
        /// responsible for a fallback in that case.
        TextureDeclHandle raw_visibility;
    };

    Outputs add_to_frame_graph(FrameGraph& fg, const PassContext& ctx, const Inputs& in);

    [[nodiscard]] bool enabled() const {
        return m_enabled;
    }

    bool m_enabled = true;

   private:
    uint64_t m_frame_counter = 0;
};

}  // namespace pts::rendering
