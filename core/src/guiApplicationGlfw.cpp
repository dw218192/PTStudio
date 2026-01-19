#include "guiApplicationImpl.h"

#if PTS_WINDOWING_GLFW
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>

#include "rendering/glfw/glfwWindowing.h"

namespace {
std::weak_ptr<spdlog::logger> g_logger;
}

namespace pts {
namespace {

struct GlfwCallbacks {
    static void click_func(GLFWwindow*, int, int, int) {
    }
    static void motion_func(GLFWwindow*, double, double) {
    }
    static void scroll_func(GLFWwindow* window, double x, double y) {
        auto* app = static_cast<GUIApplication*>(glfwGetWindowUserPointer(window));
        if (!app) {
            return;
        }
        app->on_scroll_event(x, y);
    }
    static void key_func(GLFWwindow*, int, int, int, int) {
    }
    static void error_func(int error, const char* description) {
        if (auto logger = g_logger.lock()) {
            logger->error("GLFW error: {}: {}", error, description);
        } else {
            std::cerr << "GLFW error: " << error << ": " << description << std::endl;
        }
    }
    static void framebuffer_resize_func(GLFWwindow* window, int, int) {
        auto* app = static_cast<GUIApplication*>(glfwGetWindowUserPointer(window));
        if (!app) {
            return;
        }
        app->on_framebuffer_resized();
    }
};
}  // namespace

class GlfwGuiApplicationImpl final : public GUIApplication::Impl {
   public:
    GlfwGuiApplicationImpl(GUIApplication& app, std::string_view name,
                           LoggingManager& logging_manager, unsigned width, unsigned height)
        : m_app(&app) {
        g_logger = logging_manager.get_logger_shared("GUIApplication");
        glfwSetErrorCallback(GlfwCallbacks::error_func);
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        m_window = glfwCreateWindow(width, height, name.data(), nullptr, nullptr);
        if (!m_window) {
            glfwTerminate();
            throw std::runtime_error("Failed to create window");
        }

        // set callbacks
        glfwSetWindowUserPointer(m_window, m_app);
        glfwSetMouseButtonCallback(m_window, GlfwCallbacks::click_func);
        glfwSetCursorPosCallback(m_window, GlfwCallbacks::motion_func);
        glfwSetScrollCallback(m_window, GlfwCallbacks::scroll_func);
        glfwSetKeyCallback(m_window, GlfwCallbacks::key_func);
        glfwSetFramebufferSizeCallback(m_window, GlfwCallbacks::framebuffer_resize_func);
    }

    ~GlfwGuiApplicationImpl() override {
        if (m_window) {
            glfwDestroyWindow(m_window);
        }
        glfwTerminate();
    }

    [[nodiscard]] auto create_windowing() const -> std::unique_ptr<rendering::IWindowing> override {
        return std::make_unique<GlfwWindowing>(m_window);
    }

    void poll_events() const override {
        glfwPollEvents();
    }

    [[nodiscard]] auto should_close() const -> bool override {
        return glfwWindowShouldClose(m_window) != 0;
    }

    [[nodiscard]] auto time() const noexcept -> double override {
        return glfwGetTime();
    }

    [[nodiscard]] auto window_extent() const noexcept -> glm::ivec2 override {
        int width = 0;
        int height = 0;
        glfwGetWindowSize(m_window, &width, &height);
        return {width, height};
    }

    void set_cursor_pos(float x, float y) noexcept override {
        glfwSetCursorPos(m_window, x, y);
    }

    void cursor_pos(double& x, double& y) const override {
        glfwGetCursorPos(m_window, &x, &y);
    }

    [[nodiscard]] auto middle_mouse_button() const noexcept -> int override {
        return GLFW_MOUSE_BUTTON_MIDDLE;
    }

   private:
    GUIApplication* m_app{nullptr};
    GLFWwindow* m_window{nullptr};
};
}  // namespace pts

namespace pts {
auto create_gui_application_impl(GUIApplication& app, std::string_view name,
                                 LoggingManager& logging_manager, unsigned width, unsigned height)
    -> std::unique_ptr<GUIApplication::Impl> {
    return std::make_unique<GlfwGuiApplicationImpl>(app, name, logging_manager, width, height);
}
}  // namespace pts

#endif
