#include "vulkanContext.h"

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

VulkanContext::VulkanContext(vk::Instance instance, vk::SurfaceKHR surface,
                             LoggingManager& logging_manager)
    : m_instance(instance), m_surface(surface) {
    m_logger = logging_manager.get_logger_shared("VulkanContext");
    auto devices = m_instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan physical devices found");
    }
    m_logger->info("Found {} Vulkan physical device(s)", devices.size());

    for (auto const& device : devices) {
        if (!has_extension(device, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            if (m_logger) {
                m_logger->debug("Skipping device without swapchain extension");
            }
            continue;
        }

        auto queue_families = device.getQueueFamilyProperties();
        for (uint32_t i = 0; i < queue_families.size(); ++i) {
            if (queue_families[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                if (device.getSurfaceSupportKHR(i, m_surface)) {
                    m_physical_device = device;
                    m_graphics_queue_family = i;
                    break;
                }
            }
        }
        if (m_physical_device) {
            break;
        }
    }

    if (!m_physical_device) {
        throw std::runtime_error("No suitable Vulkan device with present support found");
    }
    auto const props = m_physical_device.getProperties();
    m_logger->info("Using Vulkan device: {}", props.deviceName.data());
    m_logger->info("Graphics queue family: {}", m_graphics_queue_family);

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
