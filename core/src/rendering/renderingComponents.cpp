#include "renderingComponents.h"

#include <core/rendering/webgpuContext.h>
#include <imgui.h>
#include <imgui_impl_null.h>

#include <memory>
#include <utility>

#include "null/nullImguiRendering.h"
#include "webgpu/webgpuImguiRendering.h"

namespace spdlog {
class logger;
}

namespace pts::rendering {

auto create_imgui_components(WebGpuContext& webgpu_context, pts::rendering::IViewport& viewport,
                             pts::LoggingManager& logging_manager) -> ImGuiComponents {
    ImGuiComponents components;
    components.render_graph = nullptr;

    auto webgpu_rendering =
        create_webgpu_imgui_rendering(webgpu_context, viewport, logging_manager);
    if (webgpu_rendering) {
        components.imgui_rendering = std::move(webgpu_rendering);
        return components;
    }

    // Fallback to null rendering
    logging_manager.get_logger().warn("Using null imgui rendering as fallback");
    components.imgui_rendering = create_null_imgui_rendering(logging_manager);
    return components;
}
}  // namespace pts::rendering
