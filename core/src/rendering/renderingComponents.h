#pragma once

#include <core/loggingManager.h>
#include <core/rendering/windowing.h>

#include <memory>

#include "imguiBackend.h"
#include "renderGraph.h"

namespace pts::rendering {
struct RenderingComponents {
    std::unique_ptr<IRenderGraph> render_graph;
    std::unique_ptr<IImguiRendering> imgui_rendering;
};

[[nodiscard]] auto create_rendering_components(IWindowing& windowing, IViewport& viewport,
                                               pts::LoggingManager& logging_manager)
    -> RenderingComponents;
}  // namespace pts::rendering
