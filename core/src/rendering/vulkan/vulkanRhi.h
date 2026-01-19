#pragma once

#include <core/loggingManager.h>
#include <core/rendering/windowing.h>

#include <boost/container/stable_vector.hpp>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "../rhi.h"

namespace pts::rendering {

class VulkanRhi final : public IRhi {
   public:
    VulkanRhi(WindowingVulkanExtensions extensions, LoggingManager& logging_manager);
    ~VulkanRhi() override = default;

    VulkanRhi(const VulkanRhi&) = delete;
    VulkanRhi& operator=(const VulkanRhi&) = delete;
    VulkanRhi(VulkanRhi&&) = default;
    VulkanRhi& operator=(VulkanRhi&&) = default;

    [[nodiscard]] auto backend_type() const noexcept -> RhiBackendType override {
        return RhiBackendType::vulkan;
    }
    [[nodiscard]] auto device_handle() const noexcept -> RhiDevice override {
        return {m_device.get()};
    }
    [[nodiscard]] auto graphics_queue_handle() const noexcept -> RhiQueue override {
        return {reinterpret_cast<void*>(static_cast<VkQueue>(m_graphics_queue))};
    }
    [[nodiscard]] auto graphics_queue_family() const noexcept -> uint32_t override {
        return m_graphics_queue_family;
    }
    [[nodiscard]] auto cmd_api() const noexcept -> const PtsCmdApi* override {
        return nullptr;
    }

    [[nodiscard]] auto create_fence(bool signaled) -> RhiFence override;
    void destroy_fence(RhiFence fence) noexcept override;
    auto wait_fence(RhiFence fence, std::uint64_t timeout_ns) noexcept -> bool override;
    void reset_fence(RhiFence fence) noexcept override;

    [[nodiscard]] auto create_semaphore() -> RhiSemaphore override;
    void destroy_semaphore(RhiSemaphore semaphore) noexcept override;

    [[nodiscard]] auto fence(RhiFence handle) const -> vk::Fence;
    [[nodiscard]] auto semaphore(RhiSemaphore handle) const -> vk::Semaphore;

    [[nodiscard]] auto instance() const noexcept -> vk::Instance {
        return m_instance.get();
    }
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
    auto create_vulkan_instance(WindowingVulkanExtensions extensions) -> vk::UniqueInstance;
    void pick_physical_device();
    void create_device();

    vk::UniqueInstance m_instance;
    vk::PhysicalDevice m_physical_device{};
    vk::UniqueDevice m_device;
    vk::Queue m_graphics_queue{};
    uint32_t m_graphics_queue_family{0};
    boost::container::stable_vector<vk::Fence> m_fences;
    boost::container::stable_vector<vk::Semaphore> m_semaphores;
    std::vector<uint32_t> m_free_fence_ids;
    std::vector<uint32_t> m_free_semaphore_ids;
    std::shared_ptr<spdlog::logger> m_logger;
};

}  // namespace pts::rendering
