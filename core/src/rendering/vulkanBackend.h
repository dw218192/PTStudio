#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <core/loggingManager.h>

#include <memory>
#include <vulkan/vulkan.hpp>

#include "rhiBackend.h"

namespace pts::rendering {

class VulkanContext;
class SwapchainHost;
class RenderGraphHost;
class ImGuiVulkanPresenter;

class VulkanBackend final : public IRhiBackend {
   public:
    explicit VulkanBackend(GLFWwindow* window, LoggingManager& logging_manager);
    ~VulkanBackend() override;

    VulkanBackend(const VulkanBackend&) = delete;
    VulkanBackend& operator=(const VulkanBackend&) = delete;
    VulkanBackend(VulkanBackend&&) = delete;
    VulkanBackend& operator=(VulkanBackend&&) = delete;

    void new_frame() override;
    void render(bool framebuffer_resized) override;
    void resize_render_graph(uint32_t width, uint32_t height) override;
    void set_render_graph_current() override;
    void clear_render_graph_current() override;

    [[nodiscard]] auto render_graph_api() const noexcept -> const PtsRenderGraphApi* override;
    [[nodiscard]] auto output_texture() const noexcept -> PtsTexture override;
    [[nodiscard]] auto output_imgui_id() const noexcept -> ImTextureID override;

   private:
    void create_vulkan_instance();
    void create_surface();

    GLFWwindow* m_window{nullptr};
    LoggingManager& m_logging_manager;
    vk::UniqueInstance m_instance;
    vk::UniqueSurfaceKHR m_surface;
    std::unique_ptr<VulkanContext> m_context;
    std::unique_ptr<SwapchainHost> m_swapchain;
    std::unique_ptr<RenderGraphHost> m_render_graph;
    std::unique_ptr<ImGuiVulkanPresenter> m_presenter;
    ImTextureID m_output_id{nullptr};
    std::shared_ptr<spdlog::logger> m_logger;
};

}  // namespace pts::rendering
