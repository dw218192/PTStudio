#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

namespace pts::rendering {

class ShaderLoader;

/// Renders view-space normals, depth, and screen-space motion vectors as a
/// geometry pre-pass. Added as a child pass of any renderer via
/// add_pass<GBufferPass>(sl).
///
/// Motion vectors encode prev_uv - curr_uv (the displacement to apply to a
/// current-frame UV to sample previous-frame data). Static camera + static
/// geometry yields ~zero motion. The pass tracks previous-frame per-object
/// transforms and previous view/proj internally so callers don't have to
/// thread that state through.
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
        TextureDeclHandle motion;
    };
    Outputs add_to_frame_graph(FrameGraph& fg, const PassContext& ctx, const Inputs&);

   private:
    static constexpr uint32_t k_uniform_align = 256;

    // Previous-frame state for motion-vector reprojection. Per-slot prev
    // transforms tracked by SlotMap index; `valid` is false when the slot was
    // inactive/invisible last frame (motion = 0 for the first frame after a
    // slot turns on). Camera matrices use the same convention: first-frame
    // path picks prev = curr so motion = 0.
    struct PrevObjectState {
        glm::mat4 transform{1.0f};
        bool valid = false;
    };
    std::vector<PrevObjectState> m_prev_objects;
    glm::mat4 m_prev_view{1.0f};
    glm::mat4 m_prev_proj{1.0f};
    bool m_prev_camera_valid = false;
};

}  // namespace pts::rendering
