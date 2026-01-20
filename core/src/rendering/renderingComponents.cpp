#include "renderingComponents.h"

#include <memory>
#include <utility>

#if defined(PTS_GAPI_vulkan)
#include "vulkan/vulkanImguiRendering.h"
#include "vulkan/vulkanPresent.h"
#include "vulkan/vulkanRenderGraph.h"
#include "vulkan/vulkanRhi.h"
#elif defined(PTS_GAPI_null)
#include "null/nullImgui.h"
#include "null/nullRendering.h"
#endif

namespace pts::rendering {
auto create_rendering_components(IWindowing& windowing, IViewport& viewport,
                                 pts::LoggingManager& logging_manager) -> RenderingComponents {
#if defined(PTS_GAPI_vulkan)
    auto rhi = std::make_shared<VulkanRhi>(windowing.required_vulkan_instance_extensions(),
                                           logging_manager);
    auto present = std::make_shared<VulkanPresent>(windowing, viewport, *rhi, logging_manager);

    auto components = RenderingComponents{};
    components.render_graph = std::make_unique<VulkanRenderGraph>(rhi, logging_manager);
    components.imgui_rendering =
        std::make_unique<VulkanImguiRendering>(std::move(rhi), std::move(present), logging_manager);
    return components;
#elif defined(PTS_GAPI_null)
    static_cast<void>(windowing);
    static_cast<void>(viewport);
    auto components = RenderingComponents{};
    components.render_graph = create_null_render_graph(logging_manager);
    components.imgui_rendering = create_null_imgui_rendering(logging_manager);
    return components;
#else
#error "Unsupported GAPI"
#endif
}
}  // namespace pts::rendering
