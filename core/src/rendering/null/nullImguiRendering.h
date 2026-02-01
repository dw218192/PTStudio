#pragma once

#include <core/loggingManager.h>
#include <imgui.h>
#include <imgui_impl_null.h>

#include <memory>

#include "../imguiBackend.h"

namespace pts::rendering {
class NullImguiRendering final : public IImguiRendering {
   public:
    explicit NullImguiRendering(pts::LoggingManager& logging_manager)
        : m_logger(logging_manager.get_logger_shared("NullImguiRendering")) {
        if (!m_logger) {
            throw std::runtime_error("Failed to create logger");
        }
        if (!ImGui_ImplNullRender_Init()) {
            throw std::runtime_error("Failed to initialize ImGui null renderer");
        }
        m_logger->info("ImGui null renderer initialized");
    }

    ~NullImguiRendering() override {
        ImGui_ImplNullRender_Shutdown();
        m_logger->info("ImGui null renderer destroyed");
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
};

inline std::unique_ptr<IImguiRendering> create_null_imgui_rendering(
    pts::LoggingManager& logging_manager) {
    try {
        return std::make_unique<NullImguiRendering>(logging_manager);
    } catch (const std::runtime_error& e) {
        logging_manager.get_logger().error("Failed to create null imgui rendering: {}", e.what());
        return nullptr;
    }
}
}  // namespace pts::rendering