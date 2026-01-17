#pragma once

#include <core/loggingManager.h>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "vulkanContext.h"

struct GLFWwindow;

namespace pts::rendering {
class SwapchainHost {
   public:
    SwapchainHost(GLFWwindow* window, VulkanContext& context, LoggingManager& logging_manager);
    ~SwapchainHost();

    SwapchainHost(const SwapchainHost&) = delete;
    SwapchainHost& operator=(const SwapchainHost&) = delete;
    SwapchainHost(SwapchainHost&&) = delete;
    SwapchainHost& operator=(SwapchainHost&&) = delete;

    void resize();

    [[nodiscard]] auto swapchain() const noexcept -> vk::SwapchainKHR {
        return m_swapchain.get();
    }
    [[nodiscard]] auto format() const noexcept -> vk::Format {
        return m_format;
    }
    [[nodiscard]] auto extent() const noexcept -> vk::Extent2D {
        return m_extent;
    }
    [[nodiscard]] auto images() const noexcept -> const std::vector<vk::Image>& {
        return m_images;
    }
    [[nodiscard]] auto image_views() const noexcept -> const std::vector<vk::UniqueImageView>& {
        return m_image_views;
    }
    [[nodiscard]] auto image_count() const noexcept -> uint32_t {
        return static_cast<uint32_t>(m_images.size());
    }

    [[nodiscard]] auto acquire_next_image(vk::Semaphore semaphore, uint32_t* image_index)
        -> vk::Result;
    [[nodiscard]] auto present(vk::Semaphore wait_semaphore, uint32_t image_index) -> vk::Result;

   private:
    void create_swapchain();
    void create_image_views();
    void cleanup_swapchain();
    void recreate_swapchain();

    GLFWwindow* m_window{nullptr};
    VulkanContext& m_context;
    vk::UniqueSwapchainKHR m_swapchain;
    vk::Format m_format{vk::Format::eUndefined};
    vk::Extent2D m_extent{};
    std::vector<vk::Image> m_images;
    std::vector<vk::UniqueImageView> m_image_views;
    std::shared_ptr<spdlog::logger> m_logger;
};
}  // namespace pts::rendering
