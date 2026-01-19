#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <core/rendering/windowing.h>

#include <stdexcept>

namespace pts {
class GlfwWindowing final : public rendering::IWindowing {
   public:
    explicit GlfwWindowing(GLFWwindow* window) : m_window(window) {
    }

    [[nodiscard]] auto native_handle() const noexcept -> rendering::WindowingHandle override {
        auto handle = rendering::WindowingHandle{};
        handle.type = rendering::WindowingType::glfw;
        handle.window_handle = m_window;
        handle.platform_handle = glfwGetWin32Window(m_window);
        return handle;
    }

    [[nodiscard]] auto framebuffer_extent() const noexcept
        -> rendering::FramebufferExtent override {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(m_window, &width, &height);
        return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    }

    [[nodiscard]] auto required_vulkan_instance_extensions() const noexcept
        -> rendering::WindowingVulkanExtensions override {
        uint32_t count = 0;
        auto names = glfwGetRequiredInstanceExtensions(&count);
        return {names, count};
    }

    void wait_events() const override {
        glfwWaitEvents();
    }

   private:
    GLFWwindow* m_window{nullptr};
};
}  // namespace pts