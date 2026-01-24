#pragma once

#include <core/loggingManager.h>
#include <core/rendering/windowing.h>

struct GLFWwindow;

namespace pts::rendering {
class GlfwWindowing final : public IWindowing {
   public:
    explicit GlfwWindowing(pts::LoggingManager& logging_manager);
    ~GlfwWindowing() override;

    [[nodiscard]] std::unique_ptr<IViewport> create_viewport(const ViewportDesc& desc) override;
    [[nodiscard]] NativeViewportHandle native_handle() const noexcept override;
    [[nodiscard]] auto required_vulkan_instance_extensions() const noexcept
        -> rendering::WindowingVulkanExtensions override;
    void pump_events(PumpEventMode mode) override;

    void clear_primary_window(GLFWwindow* window) noexcept;

   private:
    GLFWwindow* m_primary_window{nullptr};
    std::shared_ptr<spdlog::logger> m_logger;
};
}  // namespace pts::rendering