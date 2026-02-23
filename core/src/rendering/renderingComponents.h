#pragma once

#include <core/loggingManager.h>

#include <memory>

#include "imguiBackend.h"

namespace pts::rendering {
class WebGpuContext;

struct ImGuiComponents {
    std::unique_ptr<IImguiRendering> imgui_rendering;
};

[[nodiscard]] auto create_imgui_components(WebGpuContext& webgpu_context,
                                           pts::rendering::IViewport& viewport,
                                           pts::LoggingManager& logging_manager) -> ImGuiComponents;
}  // namespace pts::rendering
