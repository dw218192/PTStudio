#include "renderingComponents.h"

#include <imgui.h>
#include <imgui_impl_null.h>

#include <memory>
#include <utility>

namespace spdlog {
class logger;
}

namespace pts::rendering {
namespace {
class NullImguiRendering final : public IImguiRendering {
   public:
    explicit NullImguiRendering(std::shared_ptr<spdlog::logger> logger)
        : m_logger(std::move(logger)) {
        m_initialized = ImGui_ImplNullRender_Init();
        if (m_logger) {
            if (m_initialized) {
                m_logger->info("ImGui null renderer initialized");
            } else {
                m_logger->error("ImGui null renderer failed to initialize");
            }
        }
    }

    ~NullImguiRendering() override {
        if (m_initialized) {
            ImGui_ImplNullRender_Shutdown();
        }
        if (m_logger) {
            m_logger->info("ImGui null renderer destroyed");
        }
    }

    void new_frame() override {
        ImGui_ImplNullRender_NewFrame();
    }

    void render(bool framebuffer_resized) override {
        static_cast<void>(framebuffer_resized);
        ImGui_ImplNullRender_RenderDrawData(ImGui::GetDrawData());
    }

    void resize() override {
    }

    auto set_render_output(IRenderGraph& render_graph) -> ImTextureID override {
        static_cast<void>(render_graph);
        return ImTextureID_Invalid;
    }

    void clear_render_output() override {
    }

    [[nodiscard]] auto output_id() const noexcept -> ImTextureID override {
        return ImTextureID_Invalid;
    }

   private:
    std::shared_ptr<spdlog::logger> m_logger;
    bool m_initialized{false};
};
}  // namespace

auto create_rendering_components(IWindowing& windowing, IViewport& viewport,
                                 pts::LoggingManager& logging_manager) -> RenderingComponents {
    static_cast<void>(windowing);
    static_cast<void>(viewport);
    RenderingComponents components;
    components.render_graph = nullptr;
    components.imgui_rendering =
        std::make_unique<NullImguiRendering>(logging_manager.get_logger_shared("ImGuiRendering"));
    return components;
}
}  // namespace pts::rendering
