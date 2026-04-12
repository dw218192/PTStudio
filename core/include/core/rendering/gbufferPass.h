#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/outputLayout.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>

namespace pts::rendering {

class ShaderLoader;

/// Renders view-space normals and depth as a geometry pre-pass.
/// Added as a child pass of any renderer via add_pass<GBufferPass>(sl).
class GBufferPass final : public IPass {
   public:
    using IPass::IPass;

    GBufferPass(const GBufferPass&) = delete;
    GBufferPass& operator=(const GBufferPass&) = delete;
    GBufferPass(GBufferPass&&) = delete;
    GBufferPass& operator=(GBufferPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "gbuffer";
    }
    [[nodiscard]] auto debug_targets() const noexcept
        -> std::pair<const DebugTarget*, uint32_t> override;

    struct Inputs {};
    struct Outputs {
        TextureDeclHandle depth;
        TextureDeclHandle normals;
        /// Consumer descriptor for downstream passes (depth + normals + samplers).
        DescriptorDeclHandle consumer_desc;
    };
    Outputs add_to_frame_graph(FrameGraph& fg, const PassContext& ctx, const Inputs&);

    /// Output slot declarations for the consumer bind group.
    /// Static — the slots are always the same regardless of instance state.
    /// Child passes (contactShadowPass, ssaoPass) call this to concatenate into their layouts.
    [[nodiscard]] static std::vector<OutputSlot> consumer_slots();

   private:
    static constexpr uint32_t k_uniform_align = 256;
};

}  // namespace pts::rendering
