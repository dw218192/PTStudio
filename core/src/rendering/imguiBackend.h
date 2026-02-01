#pragma once

#include <imgui.h>

#include <memory>

namespace pts {
class LoggingManager;
}

namespace pts::rendering {
class IViewport;
class IRenderGraph;

class IImguiWindowing {
   public:
    virtual ~IImguiWindowing() = default;
    virtual void new_frame() = 0;
};

class IImguiRendering {
   public:
    virtual ~IImguiRendering() = default;
    virtual void new_frame() = 0;
    virtual void render(bool framebuffer_resized) = 0;
    virtual void resize() = 0;

    virtual auto set_render_output(IRenderGraph& render_graph) -> ImTextureID = 0;
    virtual void clear_render_output() = 0;
    [[nodiscard]] virtual auto output_id() const noexcept -> ImTextureID = 0;

    // Hook for custom rendering before ImGui
    virtual void render_before_imgui() {
    }
};

struct ImguiBackendComponents {
    std::unique_ptr<IImguiWindowing> imgui_windowing;
    std::unique_ptr<IImguiRendering> imgui_rendering;
};

[[nodiscard]] auto create_imgui_windowing(IViewport& viewport, pts::LoggingManager& logging_manager)
    -> std::unique_ptr<IImguiWindowing>;

}  // namespace pts::rendering
