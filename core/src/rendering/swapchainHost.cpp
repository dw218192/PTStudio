#include <core/rendering/swapchainHost.h>

#include <GLFW/glfw3.h>

#include <algorithm>

namespace PTS::rendering {
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
    return m_context.device().acquireNextImageKHR(m_swapchain.get(), UINT64_MAX, semaphore, {},
                                                  image_index);
}

vk::Result SwapchainHost::present(vk::Semaphore wait_semaphore, uint32_t image_index) {
    auto present_info = vk::PresentInfoKHR{}
                            .setWaitSemaphores(wait_semaphore)
                            .setSwapchains(m_swapchain.get())
                            .setImageIndices(image_index);
    return m_context.queue().presentKHR(present_info);
}

void SwapchainHost::create_swapchain() {
    auto const caps = m_context.physical_device().getSurfaceCapabilitiesKHR(m_context.surface());
    auto const formats = m_context.physical_device().getSurfaceFormatsKHR(m_context.surface());
    auto const present_modes =
        m_context.physical_device().getSurfacePresentModesKHR(m_context.surface());

    auto surface_format = formats.front();
    for (auto const& fmt : formats) {
        if (fmt.format == vk::Format::eB8G8R8A8Srgb &&
            fmt.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            surface_format = fmt;
            break;
        }
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

    m_format = surface_format.format;
    m_extent = extent;

    auto swapchain_info =
        vk::SwapchainCreateInfoKHR{}
            .setSurface(m_context.surface())
            .setMinImageCount(image_count)
            .setImageFormat(m_format)
            .setImageColorSpace(surface_format.colorSpace)
            .setImageExtent(m_extent)
            .setImageArrayLayers(1)
            .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
            .setImageSharingMode(vk::SharingMode::eExclusive)
            .setPreTransform(caps.currentTransform)
            .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
            .setPresentMode(present_mode)
            .setClipped(true);

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
}  // namespace PTS::rendering

