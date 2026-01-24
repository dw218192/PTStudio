#pragma once

#include <core/rendering/graph.h>

#include <memory>

#include "rhi.h"

namespace pts {
class LoggingManager;
}

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

[[nodiscard]] auto create_render_graph(IRhi& rhi, pts::LoggingManager& logging_manager)
    -> std::unique_ptr<IRenderGraph>;
}  // namespace pts::rendering
