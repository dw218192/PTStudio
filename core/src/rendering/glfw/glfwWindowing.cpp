#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#endif

#include "glfwWindowing.h"

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <core/loggingManager.h>

#include <iostream>
#include <memory>
#include <stdexcept>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <X11/Xlib.h>
#endif

namespace pts::rendering {
namespace {
constexpr const char* k_glfw_logger_name = "GlfwWindowing";

constexpr auto k_windowing_type = WindowingType::glfw;

auto native_platform() -> NativePlatform {
#if defined(_WIN32)
    return NativePlatform::win32;
#elif defined(__linux__)
    return NativePlatform::xlib;
#else
    return NativePlatform::emscripten;
#endif
}

auto build_native_handle(GLFWwindow* window) -> NativeViewportHandle {
    auto handle = NativeViewportHandle{};
    handle.windowing = k_windowing_type;
    handle.platform = native_platform();
    handle.windowing_handle = window;
#if defined(_WIN32)
    handle.win32.hinstance = GetModuleHandle(nullptr);
    handle.win32.hwnd = glfwGetWin32Window(window);
#elif defined(__linux__)
    handle.xlib.display = glfwGetX11Display();
    handle.xlib.window = static_cast<uint64_t>(glfwGetX11Window(window));
#else
    static_cast<void>(window);
    handle.web.canvas_selector = "#canvas";
#endif
    return handle;
}

struct GlfwViewportCallbacks {
    static auto get_viewport(GLFWwindow* window) -> IViewport* {
        return static_cast<IViewport*>(glfwGetWindowUserPointer(window));
    }

    static void scroll_func(GLFWwindow* window, double x, double y) {
        auto* viewport = get_viewport(window);
        if (!viewport) {
            return;
        }
        viewport->on_scroll(x, y);
    }

    static void framebuffer_resize_func(GLFWwindow* window, int width, int height) {
        auto* viewport = get_viewport(window);
        if (!viewport) {
            return;
        }
        viewport->on_drawable_resized(
            Extent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
    }

    static void close_func(GLFWwindow* window) {
        auto* viewport = get_viewport(window);
        if (!viewport) {
            return;
        }
        viewport->on_close_requested();
    }
};

struct GlfwErrorCallback {
    static void error_func(int error, const char* description) {
        pts::log_or_cerr(k_glfw_logger_name, pts::LogLevel::Error, "GLFW error: {}: {}", error,
                         description);
    }
};

class GlfwViewport final : public IViewport {
   public:
    GlfwViewport(GLFWwindow* window, GlfwWindowing* owner) : m_window(window), m_owner(owner) {
        glfwSetWindowUserPointer(m_window, this);
        glfwSetScrollCallback(m_window, GlfwViewportCallbacks::scroll_func);
        glfwSetFramebufferSizeCallback(m_window, GlfwViewportCallbacks::framebuffer_resize_func);
        glfwSetWindowCloseCallback(m_window, GlfwViewportCallbacks::close_func);
    }

    ~GlfwViewport() override {
        if (m_owner) {
            m_owner->clear_primary_window(m_window);
        }
        if (m_window) {
            glfwDestroyWindow(m_window);
        }
    }

    [[nodiscard]] auto native_handle() const noexcept -> NativeViewportHandle override {
        return build_native_handle(m_window);
    }

    [[nodiscard]] auto drawable_extent() const noexcept -> Extent2D override {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(m_window, &width, &height);
        return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    }

    [[nodiscard]] auto logical_extent() const noexcept -> Extent2D override {
        int width = 0;
        int height = 0;
        glfwGetWindowSize(m_window, &width, &height);
        return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    }

    [[nodiscard]] auto content_scale() const noexcept -> float override {
        float xscale = 1.0f;
        float yscale = 1.0f;
        glfwGetWindowContentScale(m_window, &xscale, &yscale);
        return (xscale + yscale) * 0.5f;
    }

    [[nodiscard]] auto should_close() const noexcept -> bool override {
        return glfwWindowShouldClose(m_window) != 0;
    }

    void request_close() noexcept override {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }

    void set_title(const char* utf8) override {
        if (utf8) {
            glfwSetWindowTitle(m_window, utf8);
        }
    }

    void set_visible(bool visible) override {
        if (visible) {
            glfwShowWindow(m_window);
        } else {
            glfwHideWindow(m_window);
        }
    }

    void set_cursor_pos(double x, double y) noexcept override {
        glfwSetCursorPos(m_window, x, y);
    }

   private:
    GLFWwindow* m_window{nullptr};
    GlfwWindowing* m_owner{nullptr};
};
}  // namespace

GlfwWindowing::GlfwWindowing(pts::LoggingManager& logging_manager) {
    m_logger = logging_manager.get_logger_shared(k_glfw_logger_name);

    glfwSetErrorCallback(GlfwErrorCallback::error_func);
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    m_logger->info("GLFW initialized");
}

GlfwWindowing::~GlfwWindowing() {
    glfwTerminate();
    if (m_logger) {
        m_logger->info("GLFW terminated");
    }
}

auto GlfwWindowing::create_viewport(const ViewportDesc& desc) -> std::unique_ptr<IViewport> {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, desc.resizable ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_VISIBLE, desc.visible ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_DECORATED, desc.decorated ? GLFW_TRUE : GLFW_FALSE);
#ifdef GLFW_SCALE_TO_MONITOR
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, desc.high_dpi ? GLFW_TRUE : GLFW_FALSE);
#endif
#ifdef GLFW_COCOA_RETINA_FRAMEBUFFER
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, desc.high_dpi ? GLFW_TRUE : GLFW_FALSE);
#endif

    auto* window = glfwCreateWindow(static_cast<int>(desc.width), static_cast<int>(desc.height),
                                    desc.title ? desc.title : "", nullptr, nullptr);
    if (!window) {
        throw std::runtime_error("Failed to create GLFW window");
    }

    if (!m_primary_window) {
        m_primary_window = window;
    }

    return std::make_unique<GlfwViewport>(window, this);
}

auto GlfwWindowing::native_handle() const noexcept -> NativeViewportHandle {
    if (!m_primary_window) {
        auto handle = NativeViewportHandle{};
        handle.windowing = k_windowing_type;
        handle.platform = native_platform();
        return handle;
    }
    return build_native_handle(m_primary_window);
}

void GlfwWindowing::pump_events(PumpEventMode mode) {
    if (mode == PumpEventMode::Wait) {
        glfwWaitEvents();
    } else {
        glfwPollEvents();
    }
}

void GlfwWindowing::clear_primary_window(GLFWwindow* window) noexcept {
    if (m_primary_window == window) {
        m_primary_window = nullptr;
    }
}

auto create_windowing(pts::LoggingManager& logging_manager) -> std::unique_ptr<IWindowing> {
    return std::make_unique<GlfwWindowing>(logging_manager);
}
}  // namespace pts::rendering
