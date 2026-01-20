#pragma once

#include <core/loggingManager.h>

#include <memory>

#include "../imguiBackend.h"

namespace pts::rendering {
[[nodiscard]] auto create_null_imgui_rendering(pts::LoggingManager& logging_manager)
    -> std::unique_ptr<IImguiRendering>;
}  // namespace pts::rendering
