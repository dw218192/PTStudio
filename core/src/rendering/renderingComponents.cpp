#include "renderingComponents.h"

#include <memory>
#include <utility>

#include "vulkan/vulkanImguiRendering.h"
#include "vulkan/vulkanPresent.h"
#include "vulkan/vulkanRenderGraph.h"
#include "vulkan/vulkanRhi.h"

namespace pts::rendering {
auto create_rendering_components(IWindowing& windowing, IViewport& viewport,
                                 pts::LoggingManager& logging_manager) -> RenderingComponents {
    auto rhi = std::make_shared<VulkanRhi>(windowing.required_vulkan_instance_extensions(),
                                           logging_manager);
    auto present = std::make_shared<VulkanPresent>(windowing, viewport, *rhi, logging_manager);

    auto components = RenderingComponents{};
    components.render_graph = std::make_unique<VulkanRenderGraph>(rhi, logging_manager);
    components.imgui_rendering =
        std::make_unique<VulkanImguiRendering>(std::move(rhi), std::move(present), logging_manager);
    return components;
}
}  // namespace pts::rendering
