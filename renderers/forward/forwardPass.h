#pragma once

#include <core/rendering/renderer.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <string_view>

namespace pts::editor {

class ForwardPass final : public rendering::IRenderer {
   public:
    explicit ForwardPass(const rendering::ShaderLoader& sl);

    ForwardPass(const ForwardPass&) = delete;
    ForwardPass& operator=(const ForwardPass&) = delete;
    ForwardPass(ForwardPass&&) = delete;
    ForwardPass& operator=(ForwardPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto renderer_debug_targets() const noexcept
        -> std::pair<const DebugTarget*, uint32_t> override;

    HdrOutputs do_add_to_frame_graph(rendering::FrameGraph& fg,
                                     const rendering::PassContext& ctx) override;
    static constexpr uint32_t k_uniform_align = 256;
};

}  // namespace pts::editor
