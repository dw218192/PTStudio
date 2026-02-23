#include "renderingComponents.h"

#include <core/rendering/webgpuContext.h>
#include <imgui.h>

#include <memory>
#include <stdexcept>
#include <utility>

#include "webgpu/webgpuImguiRendering.h"

namespace spdlog {
class logger;
}

namespace pts::rendering {

auto create_imgui_components(WebGpuContext& webgpu_context, pts::rendering::IViewport& viewport,
                             pts::LoggingManager& logging_manager) -> ImGuiComponents {
    ImGuiComponents components;

    auto webgpu_rendering =
        create_webgpu_imgui_rendering(webgpu_context, viewport, logging_manager);
    if (!webgpu_rendering) {
        throw std::runtime_error("Failed to create WebGPU imgui rendering");
    }
    components.imgui_rendering = std::move(webgpu_rendering);
    return components;
}
}  // namespace pts::rendering
