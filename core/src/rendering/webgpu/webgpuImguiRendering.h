#pragma once

#include <core/loggingManager.h>
#include <core/rendering/webgpuContext.h>

#include <memory>

#include "../imguiBackend.h"

namespace spdlog {
class logger;
}

namespace pts::rendering {

class WebGpuImguiRendering final : public IImguiRendering {
   public:
    WebGpuImguiRendering(WebGpuContext& context, IViewport& viewport,
                         pts::LoggingManager& logging_manager);
    ~WebGpuImguiRendering() override;

    void new_frame() override;
    void render(bool framebuffer_resized) override;
    void resize() override;
    auto set_render_output(IRenderGraph& render_graph) -> ImTextureID override;
    void clear_render_output() override;
    [[nodiscard]] auto output_id() const noexcept -> ImTextureID override;

    void render_before_imgui() override;

   private:
    WebGpuContext* m_context;
    IViewport* m_viewport;
    std::shared_ptr<spdlog::logger> m_logger;
    Extent2D m_extent{0, 0};
};

std::unique_ptr<IImguiRendering> create_webgpu_imgui_rendering(
    WebGpuContext& context, IViewport& viewport, pts::LoggingManager& logging_manager);
}  // namespace pts::rendering