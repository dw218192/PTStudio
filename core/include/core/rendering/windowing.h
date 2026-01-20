#pragma once

#include <core/signal.h>

#include <cstdint>
#include <memory>

namespace pts {
class LoggingManager;
}

namespace pts::rendering {
enum class WindowingType : std::uint32_t {
    glfw = 1,
};
enum class NativePlatform : std::uint32_t { win32, xlib, emscripten };

struct NativeViewportHandle {
    WindowingType windowing;
    NativePlatform platform;
    void* windowing_handle{nullptr};

    union {
        struct {
            void* hinstance;
            void* hwnd;
        } win32;  // HINSTANCE, HWND
        struct {
            void* display;
            uint64_t window;
        } xlib;  // Display*, Window
        struct {
            const char* canvas_selector;
        } web;  // "#canvas" / "canvas"
    };
};
enum class PumpEventMode {
    Wait,
    Poll,
};

struct WindowingVulkanExtensions {
    const char* const* names{nullptr};
    std::uint32_t count{0};
};
struct Extent2D {
    uint32_t w, h;
};

struct ViewportDesc {
    const char* title;
    uint32_t width;
    uint32_t height;
    bool resizable;
    bool visible;
    bool decorated;
    bool high_dpi;
};

class IViewport {
   public:
    virtual ~IViewport() = default;

    [[nodiscard]] virtual NativeViewportHandle native_handle() const noexcept = 0;

    /**
     * @brief What you render to (swapchain / surface size). HiDPI-safe
     */
    [[nodiscard]] virtual Extent2D drawable_extent() const noexcept = 0;

    /**
     * @brief logical extent for UI/layout
     */
    [[nodiscard]] virtual Extent2D logical_extent() const noexcept = 0;

    /**
     * @brief scale between drawable and logical extent
     */
    [[nodiscard]] virtual float content_scale() const noexcept = 0;

    /**
     * @brief whether the window should close
     */
    [[nodiscard]] virtual bool should_close() const noexcept = 0;

    /**
     * @brief request the window to close
     */
    virtual void request_close() noexcept = 0;

    /**
     * @brief set the title of the window
     */
    virtual void set_title(const char* utf8) = 0;

    /**
     * @brief set the visibility of the window
     */
    virtual void set_visible(bool v) = 0;

    /**
     * @brief set the cursor position
     */
    virtual void set_cursor_pos(double x, double y) noexcept = 0;

    /**
     * @brief signal emitted when the drawable extent changes. Takes the new extent as parameter.
     */
    Signal<void(Extent2D new_extent)> on_drawable_resized;

    /**
     * @brief signal emitted when the window is closed
     */
    Signal<void()> on_close_requested;

    /**
     * @brief signal emitted when the window is scrolled. Takes the scroll delta as parameters.
     */
    Signal<void(double dx, double dy)> on_scroll;
};

class IWindowing {
   public:
    virtual ~IWindowing() = default;

    /**
     * @brief Create a new viewport.
     */
    [[nodiscard]] virtual std::unique_ptr<IViewport> create_viewport(const ViewportDesc& desc) = 0;

    /**
     * @brief Get the native handle of the windowing system.
     */
    [[nodiscard]] virtual NativeViewportHandle native_handle() const noexcept = 0;

    /**
     * @brief Pump events from the windowing system. Blocks the current thread if mode is Wait.
     */
    virtual void pump_events(PumpEventMode mode) = 0;

    /**
     * @brief Get the required Vulkan instance extensions for the windowing system. If the windowing
     * system does not support Vulkan, returns an empty array.
     */
    [[nodiscard]] virtual WindowingVulkanExtensions required_vulkan_instance_extensions()
        const noexcept = 0;
};

[[nodiscard]] auto create_windowing(pts::LoggingManager& logging_manager)
    -> std::unique_ptr<IWindowing>;

}  // namespace pts::rendering
