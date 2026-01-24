#pragma once

#include <core/rendering/graph.h>

#include <memory>

namespace pts::rendering {

class IRenderGraph {
   public:
    virtual ~IRenderGraph() = default;

    virtual void resize(uint32_t width, uint32_t height) = 0;
    virtual void set_current() = 0;
    virtual void clear_current() = 0;
    [[nodiscard]] virtual auto output_texture() const noexcept -> PtsTexture = 0;
    [[nodiscard]] virtual auto api() const noexcept -> const PtsRenderGraphApi* = 0;
};

}  // namespace pts::rendering
