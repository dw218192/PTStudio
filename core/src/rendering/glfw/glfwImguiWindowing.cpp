#include "glfwImguiWindowing.h"

#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>

#include <memory>
#include <stdexcept>

namespace pts::rendering {
namespace {
GLFWwindow* get_glfw_window(const IViewport& viewport) {
    auto handle = viewport.native_handle();
    if (handle.windowing != WindowingType::glfw || !handle.windowing_handle) {
        throw std::runtime_error("ImGui windowing requires a GLFW window handle");
    }
    return static_cast<GLFWwindow*>(handle.windowing_handle);
}
}  // namespace

GlfwImguiWindowing::GlfwImguiWindowing(IViewport& viewport, pts::LoggingManager& logging_manager) {
    m_logger = logging_manager.get_logger_shared("ImGuiWindowing");
    auto* window = get_glfw_window(viewport);
    ImGui_ImplGlfw_InitForVulkan(window, true);
    m_initialized = true;
    m_logger->info("ImGui GLFW windowing initialized");
}

GlfwImguiWindowing::~GlfwImguiWindowing() {
    if (m_initialized) {
        ImGui_ImplGlfw_Shutdown();
    }
    if (m_logger) {
        m_logger->info("ImGui GLFW windowing destroyed");
    }
}

void GlfwImguiWindowing::new_frame() {
    ImGui_ImplGlfw_NewFrame();
}

auto create_imgui_windowing(IViewport& viewport, pts::LoggingManager& logging_manager)
    -> std::unique_ptr<IImguiWindowing> {
    return std::make_unique<GlfwImguiWindowing>(viewport, logging_manager);
}
}  // namespace pts::rendering
