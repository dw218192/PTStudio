#pragma once

#include <vulkan/vulkan.hpp>

namespace PTS::Editor {
class VulkanContext {
  public:
    VulkanContext(vk::Instance instance, vk::SurfaceKHR surface);
    ~VulkanContext() = default;

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&&) = default;
    VulkanContext& operator=(VulkanContext&&) = default;

    [[nodiscard]] auto physical_device() const noexcept -> vk::PhysicalDevice {
        return m_physical_device;
    }
    [[nodiscard]] auto device() const noexcept -> vk::Device {
        return m_device.get();
    }
    [[nodiscard]] auto queue() const noexcept -> vk::Queue {
        return m_graphics_queue;
    }
    [[nodiscard]] auto queue_family() const noexcept -> uint32_t {
        return m_graphics_queue_family;
    }

  private:
    vk::Instance m_instance{};
    vk::SurfaceKHR m_surface{};
    vk::PhysicalDevice m_physical_device{};
    vk::UniqueDevice m_device;
    vk::Queue m_graphics_queue{};
    uint32_t m_graphics_queue_family{0};
};
}  // namespace PTS::Editor

