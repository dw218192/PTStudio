#include <core/rendering/vulkanContext.h>

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

VulkanContext::VulkanContext(vk::Instance instance, vk::SurfaceKHR surface)
    : m_instance(instance), m_surface(surface) {
    auto devices = m_instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan physical devices found");
    }

    for (auto const& device : devices) {
        if (!has_extension(device, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
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
}
}  // namespace pts::rendering
