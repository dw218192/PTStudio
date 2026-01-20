#pragma once

#include <core/loggingManager.h>

#include "../renderGraph.h"

#include <memory>

namespace pts::rendering {
[[nodiscard]] auto create_null_render_graph(pts::LoggingManager& logging_manager)
    -> std::unique_ptr<IRenderGraph>;
}  // namespace pts::rendering

