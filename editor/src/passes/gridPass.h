#pragma once

#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/webgpu.h>

#include <string_view>

namespace pts::editor {

class GridPass final : public rendering::IPass {
   public:
    using IPass::IPass;

    GridPass(const GridPass&) = delete;
    GridPass& operator=(const GridPass&) = delete;
    GridPass(GridPass&&) = delete;
    GridPass& operator=(GridPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;

    void render(rendering::FrameGraph& fg, const rendering::PassContext& ctx,
                rendering::TextureDeclHandle color, rendering::TextureDeclHandle depth);
};

}  // namespace pts::editor
