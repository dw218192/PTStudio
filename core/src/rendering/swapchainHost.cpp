#include "swapchainHost.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <optional>
#include <stdexcept>

namespace pts::rendering {
SwapchainHost::SwapchainHost(GLFWwindow* window, VulkanContext& context)
    : m_window(window), m_context(context) {
    create_swapchain();
    create_image_views();
}

SwapchainHost::~SwapchainHost() {
    cleanup_swapchain();
}

void SwapchainHost::resize() {
    recreate_swapchain();
}

vk::Result SwapchainHost::acquire_next_image(vk::Semaphore semaphore, uint32_t* image_index) {
    try {
        return m_context.device().acquireNextImageKHR(m_swapchain.get(), UINT64_MAX, semaphore, {},
                                                      image_index);
    } catch (vk::OutOfDateKHRError const&) {
        return vk::Result::eErrorOutOfDateKHR;
    } catch (vk::SystemError const& err) {
        return static_cast<vk::Result>(err.code().value());
    }
}

vk::Result SwapchainHost::present(vk::Semaphore wait_semaphore, uint32_t image_index) {
    auto present_info = vk::PresentInfoKHR{}
                            .setWaitSemaphores(wait_semaphore)
                            .setSwapchains(m_swapchain.get())
                            .setImageIndices(image_index);
    try {
        return m_context.queue().presentKHR(present_info);
    } catch (vk::OutOfDateKHRError const&) {
        return vk::Result::eErrorOutOfDateKHR;
    } catch (vk::SystemError const& err) {
        return static_cast<vk::Result>(err.code().value());
    }
}

void SwapchainHost::create_swapchain() {
    auto const caps = m_context.physical_device().getSurfaceCapabilitiesKHR(m_context.surface());
    auto const formats = m_context.physical_device().getSurfaceFormatsKHR(m_context.surface());
    auto const present_modes =
        m_context.physical_device().getSurfacePresentModesKHR(m_context.surface());

    // Some drivers reject SRGB swapchain formats when storage usage is implied.
    // Require formats that support the usage bits we rely on.
    auto const format_usage_check = vk::ImageUsageFlagBits::eColorAttachment |
                                    vk::ImageUsageFlagBits::eTransferSrc |
                                    vk::ImageUsageFlagBits::eStorage;

    auto format_supports_usage = [&](vk::Format format) {
        try {
            static_cast<void>(m_context.physical_device().getImageFormatProperties(
                format, vk::ImageType::e2D, vk::ImageTiling::eOptimal, format_usage_check, {}));
            return true;
        } catch (vk::SystemError const& err) {
            return false;
        }
    };

    // Prefer SRGB first, then UNORM, then any supported format.
    auto surface_format = std::optional<vk::SurfaceFormatKHR>{};
    for (auto const& fmt : formats) {
        if (fmt.colorSpace != vk::ColorSpaceKHR::eSrgbNonlinear) {
            continue;
        }
        if (fmt.format == vk::Format::eB8G8R8A8Srgb && format_supports_usage(fmt.format)) {
            surface_format = fmt;
            break;
        }
        if (fmt.format == vk::Format::eB8G8R8A8Unorm && format_supports_usage(fmt.format) &&
            !surface_format) {
            surface_format = fmt;
        }
    }
    if (!surface_format) {
        // No SRGB/UNORM candidate matched; accept the first supported format.
        for (auto const& fmt : formats) {
            if (format_supports_usage(fmt.format)) {
                surface_format = fmt;
                break;
            }
        }
    }
    if (!surface_format) {
        // Hard fail to avoid creating an invalid swapchain.
        throw std::runtime_error(
            "No swapchain surface format supports required usage for the current device");
    }

    auto present_mode = vk::PresentModeKHR::eFifo;
    for (auto const& mode : present_modes) {
        if (mode == vk::PresentModeKHR::eMailbox) {
            present_mode = mode;
            break;
        }
    }

    auto extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(m_window, &width, &height);
        extent.width = std::clamp(static_cast<uint32_t>(width), caps.minImageExtent.width,
                                  caps.maxImageExtent.width);
        extent.height = std::clamp(static_cast<uint32_t>(height), caps.minImageExtent.height,
                                   caps.maxImageExtent.height);
    }

    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
        image_count = caps.maxImageCount;
    }

    m_format = surface_format->format;
    m_extent = extent;

    auto image_usage = vk::ImageUsageFlags{vk::ImageUsageFlagBits::eColorAttachment};
    if ((caps.supportedUsageFlags & image_usage) != image_usage) {
        image_usage = caps.supportedUsageFlags & image_usage;
    }

    auto swapchain_info = vk::SwapchainCreateInfoKHR{};
    swapchain_info.surface = m_context.surface();
    swapchain_info.minImageCount = image_count;
    swapchain_info.imageFormat = m_format;
    swapchain_info.imageColorSpace = surface_format->colorSpace;
    swapchain_info.imageExtent = m_extent;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = image_usage;
    swapchain_info.imageSharingMode = vk::SharingMode::eExclusive;
    swapchain_info.preTransform = caps.currentTransform;
    swapchain_info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    swapchain_info.presentMode = present_mode;
    swapchain_info.clipped = true;
    m_swapchain = m_context.device().createSwapchainKHRUnique(swapchain_info);
    m_images = m_context.device().getSwapchainImagesKHR(m_swapchain.get());
}

void SwapchainHost::create_image_views() {
    m_image_views.clear();
    m_image_views.reserve(m_images.size());
    for (auto const& image : m_images) {
        auto view = m_context.device().createImageViewUnique(
            vk::ImageViewCreateInfo{}
                .setImage(image)
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(m_format)
                .setSubresourceRange(vk::ImageSubresourceRange{}
                                         .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                         .setBaseMipLevel(0)
                                         .setLevelCount(1)
                                         .setBaseArrayLayer(0)
                                         .setLayerCount(1)));
        m_image_views.emplace_back(std::move(view));
    }
}

void SwapchainHost::cleanup_swapchain() {
    if (!m_swapchain) {
        return;
    }
    m_context.device().waitIdle();
    m_image_views.clear();
    m_swapchain.reset();
    m_images.clear();
}

void SwapchainHost::recreate_swapchain() {
    int width = 0;
    int height = 0;
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(m_window, &width, &height);
        glfwWaitEvents();
    }
    cleanup_swapchain();
    create_swapchain();
    create_image_views();
}
}  // namespace pts::rendering
