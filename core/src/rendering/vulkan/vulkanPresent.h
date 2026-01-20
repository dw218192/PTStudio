#pragma once

#include <core/loggingManager.h>
#include <core/rendering/windowing.h>

#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "../present.h"
#include "vulkanRhi.h"

namespace pts::rendering {

class VulkanPresent final : public IPresent {
   public:
    VulkanPresent(IWindowing& windowing, IViewport& viewport, VulkanRhi& rhi,
                  pts::LoggingManager& logging_manager);
    ~VulkanPresent() override;

    VulkanPresent(const VulkanPresent&) = delete;
    VulkanPresent& operator=(const VulkanPresent&) = delete;
    VulkanPresent(VulkanPresent&&) = delete;
    VulkanPresent& operator=(VulkanPresent&&) = delete;

    [[nodiscard]] auto acquire_next_backbuffer(RhiSemaphore signal_semaphore,
                                               uint32_t* index) -> PresentStatus override;
    [[nodiscard]] auto present_backbuffer(uint32_t index,
                                          RhiSemaphore wait_semaphore) -> PresentStatus override;
    void recreate_swapchain() override;
    [[nodiscard]] auto framebuffer_extent() const noexcept -> Extent2D override;

    [[nodiscard]] auto format() const noexcept -> vk::Format {
        return m_format;
    }
    [[nodiscard]] auto extent() const noexcept -> vk::Extent2D {
        return m_extent;
    }
    [[nodiscard]] auto image_views() const noexcept -> const std::vector<vk::UniqueImageView>& {
        return m_image_views;
    }
    [[nodiscard]] auto image_count() const noexcept -> uint32_t {
        return static_cast<uint32_t>(m_images.size());
    }
    [[nodiscard]] auto acquire_next_image(vk::Semaphore semaphore,
                                          uint32_t* image_index) -> vk::Result;
    [[nodiscard]] auto present(vk::Semaphore wait_semaphore, uint32_t image_index) -> vk::Result;
    void resize_swapchain();

   private:
    void create_surface();
    void create_swapchain();
    void create_image_views();
    void cleanup_swapchain();
    void do_recreate_swapchain();

    IWindowing& m_windowing;
    IViewport& m_viewport;
    VulkanRhi& m_rhi;
    vk::UniqueSurfaceKHR m_surface;
    vk::UniqueSwapchainKHR m_swapchain;
    vk::Format m_format{vk::Format::eUndefined};
    vk::Extent2D m_extent{};
    std::vector<vk::Image> m_images;
    std::vector<vk::UniqueImageView> m_image_views;
    std::shared_ptr<spdlog::logger> m_logger;
};

}  // namespace pts::rendering
