#pragma once

#include <core/rendering/renderer.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <string_view>

namespace pts::editor {

class WireframePass final : public rendering::IRenderer {
   public:
    using IRenderer::IRenderer;

    WireframePass(const WireframePass&) = delete;
    WireframePass& operator=(const WireframePass&) = delete;
    WireframePass(WireframePass&&) = delete;
    WireframePass& operator=(WireframePass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;

    HdrOutputs do_add_to_frame_graph(rendering::FrameGraph& fg,
                                     const rendering::PassContext& ctx) override;

    static constexpr uint32_t k_uniform_align = 256;
};

}  // namespace pts::editor
