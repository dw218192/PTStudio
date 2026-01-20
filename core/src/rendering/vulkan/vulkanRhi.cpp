#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(__linux__)
#define VK_USE_PLATFORM_XLIB_KHR
#endif
#include "vulkanRhi.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace pts::rendering {

namespace {
bool has_extension(vk::PhysicalDevice device, const char* name) {
    auto props = device.enumerateDeviceExtensionProperties();
    for (auto const& prop : props) {
        if (std::strcmp(prop.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}
}  // namespace

VulkanRhi::VulkanRhi(WindowingVulkanExtensions extensions, pts::LoggingManager& logging_manager) {
    m_logger = logging_manager.get_logger_shared("VulkanRhi");
    m_logger->info("Vulkan RHI initializing");
    m_instance = create_vulkan_instance(extensions);
    pick_physical_device();
    create_device();
    m_logger->info("Vulkan RHI initialized");
}

auto VulkanRhi::create_fence(bool signaled) -> RhiFence {
    auto info = vk::FenceCreateInfo{};
    if (signaled) {
        info.setFlags(vk::FenceCreateFlagBits::eSignaled);
    }
    auto fence = m_device->createFence(info);
    if (!m_free_fence_ids.empty()) {
        auto const index = m_free_fence_ids.back();
        m_free_fence_ids.pop_back();
        m_fences[index] = fence;
        return {reinterpret_cast<void*>(static_cast<std::uintptr_t>(index + 1))};
    }
    m_fences.push_back(fence);
    return {reinterpret_cast<void*>(static_cast<std::uintptr_t>(m_fences.size()))};
}

void VulkanRhi::destroy_fence(RhiFence fence) noexcept {
    if (!fence.handle) {
        return;
    }
    auto index = reinterpret_cast<std::uintptr_t>(fence.handle) - 1;
    if (index >= m_fences.size()) {
        return;
    }
    auto& handle = m_fences[index];
    if (!handle) {
        return;
    }
    m_device->destroyFence(handle);
    handle = vk::Fence{};
    m_free_fence_ids.push_back(static_cast<uint32_t>(index));
}

auto VulkanRhi::wait_fence(RhiFence fence, std::uint64_t timeout_ns) noexcept -> bool {
    if (!fence.handle) {
        return false;
    }
    auto index = reinterpret_cast<std::uintptr_t>(fence.handle) - 1;
    if (index >= m_fences.size()) {
        return false;
    }
    auto& handle = m_fences[index];
    if (!handle) {
        return false;
    }
    auto result = m_device->waitForFences(handle, true, timeout_ns);
    return result == vk::Result::eSuccess;
}

void VulkanRhi::reset_fence(RhiFence fence) noexcept {
    if (!fence.handle) {
        return;
    }
    auto index = reinterpret_cast<std::uintptr_t>(fence.handle) - 1;
    if (index >= m_fences.size()) {
        return;
    }
    auto& handle = m_fences[index];
    if (!handle) {
        return;
    }
    m_device->resetFences(handle);
}

auto VulkanRhi::create_semaphore() -> RhiSemaphore {
    auto semaphore = m_device->createSemaphore(vk::SemaphoreCreateInfo{});
    if (!m_free_semaphore_ids.empty()) {
        auto const index = m_free_semaphore_ids.back();
        m_free_semaphore_ids.pop_back();
        m_semaphores[index] = semaphore;
        return {reinterpret_cast<void*>(static_cast<std::uintptr_t>(index + 1))};
    }
    m_semaphores.push_back(semaphore);
    return {reinterpret_cast<void*>(static_cast<std::uintptr_t>(m_semaphores.size()))};
}

void VulkanRhi::destroy_semaphore(RhiSemaphore semaphore) noexcept {
    if (!semaphore.handle) {
        return;
    }
    auto index = reinterpret_cast<std::uintptr_t>(semaphore.handle) - 1;
    if (index >= m_semaphores.size()) {
        return;
    }
    auto& handle = m_semaphores[index];
    if (!handle) {
        return;
    }
    m_device->destroySemaphore(handle);
    handle = vk::Semaphore{};
    m_free_semaphore_ids.push_back(static_cast<uint32_t>(index));
}

auto VulkanRhi::fence(RhiFence handle) const -> vk::Fence {
    if (!handle.handle) {
        return {};
    }
    auto index = reinterpret_cast<std::uintptr_t>(handle.handle) - 1;
    if (index >= m_fences.size()) {
        return {};
    }
    return m_fences[index];
}

auto VulkanRhi::semaphore(RhiSemaphore handle) const -> vk::Semaphore {
    if (!handle.handle) {
        return {};
    }
    auto index = reinterpret_cast<std::uintptr_t>(handle.handle) - 1;
    if (index >= m_semaphores.size()) {
        return {};
    }
    return m_semaphores[index];
}

auto VulkanRhi::create_vulkan_instance(WindowingVulkanExtensions extensions) -> vk::UniqueInstance {
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
    auto instance_extensions = std::vector<char const*>{};
    instance_extensions.reserve(extensions.count + 2);
    for (uint32_t i = 0; i < extensions.count; ++i) {
        instance_extensions.push_back(extensions.names[i]);
    }
    instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    auto add_extension = [&](char const* name) {
        if (std::find(instance_extensions.begin(), instance_extensions.end(), name) ==
            instance_extensions.end()) {
            instance_extensions.push_back(name);
        }
    };

#if defined(_WIN32)
    add_extension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(__linux__)
    add_extension(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#endif

    auto instance_flags = vk::InstanceCreateFlags{};
#ifdef VK_KHR_portability_enumeration
    instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instance_flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

    auto layers = std::vector<char const*>{};
#ifndef NDEBUG
    layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    auto const app_info = vk::ApplicationInfo{"PTS Editor", VK_MAKE_VERSION(1, 0, 0), "PTS",
                                              VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_2};

    auto instance = vk::createInstanceUnique(
        vk::InstanceCreateInfo{instance_flags, &app_info, layers, instance_extensions});
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());
    m_logger->info("Vulkan instance created (extensions={}, layers={})", instance_extensions.size(),
                   layers.size());
    return instance;
}

void VulkanRhi::pick_physical_device() {
    auto devices = m_instance->enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan physical devices found");
    }
    m_logger->info("Found {} Vulkan physical device(s)", devices.size());

    for (auto const& device : devices) {
        if (!has_extension(device, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            m_logger->debug("Skipping device without swapchain extension");
            continue;
        }
        auto queue_families = device.getQueueFamilyProperties();
        for (uint32_t i = 0; i < queue_families.size(); ++i) {
            if (queue_families[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                m_physical_device = device;
                m_graphics_queue_family = i;
                break;
            }
        }
        if (m_physical_device) {
            break;
        }
    }

    if (!m_physical_device) {
        throw std::runtime_error("No suitable Vulkan device with graphics support found");
    }
    auto const props = m_physical_device.getProperties();
    m_logger->info("Using Vulkan device: {}", props.deviceName.data());
    m_logger->info("Graphics queue family: {}", m_graphics_queue_family);
}

void VulkanRhi::create_device() {
    float priority = 1.0f;
    auto queue_info = vk::DeviceQueueCreateInfo{}
                          .setQueueFamilyIndex(m_graphics_queue_family)
                          .setQueueCount(1)
                          .setQueuePriorities(priority);

    auto device_exts = std::vector<char const*>{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    auto device_info = vk::DeviceCreateInfo{}
                           .setQueueCreateInfos(queue_info)
                           .setPEnabledExtensionNames(device_exts);

    m_device = m_physical_device.createDeviceUnique(device_info);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(m_device.get());
    m_graphics_queue = m_device->getQueue(m_graphics_queue_family, 0);
    m_logger->info("Vulkan device created");
}

}  // namespace pts::rendering
