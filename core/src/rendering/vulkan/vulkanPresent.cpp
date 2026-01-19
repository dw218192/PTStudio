#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(__linux__)
#define VK_USE_PLATFORM_XLIB_KHR
#else
#error "Unsupported host platform for Vulkan present"
#endif

#include "vulkanPresent.h"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <X11/Xlib.h>
#endif

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace pts::rendering {
namespace {
auto map_present_result(vk::Result result) -> PresentStatus {
    switch (result) {
        case vk::Result::eSuccess:
            return PresentStatus::ok;
        case vk::Result::eSuboptimalKHR:
            return PresentStatus::suboptimal;
        case vk::Result::eErrorOutOfDateKHR:
            return PresentStatus::out_of_date;
        default:
            return PresentStatus::error;
    }
}
}  // namespace

VulkanPresent::VulkanPresent(IWindowing& windowing, VulkanRhi& rhi, LoggingManager& logging_manager)
    : m_windowing(windowing), m_rhi(rhi) {
    m_logger = logging_manager.get_logger_shared("VulkanPresent");
    create_surface();
    if (!m_rhi.physical_device().getSurfaceSupportKHR(m_rhi.queue_family(), m_surface.get())) {
        throw std::runtime_error("Selected Vulkan queue family does not support present");
    }
    m_logger->info("VulkanPresent created");
    create_swapchain();
    create_image_views();
}

VulkanPresent::~VulkanPresent() {
    cleanup_swapchain();
    if (m_logger) {
        m_logger->info("VulkanPresent destroyed");
    }
}

auto VulkanPresent::acquire_next_backbuffer(RhiSemaphore signal_semaphore, uint32_t* index)
    -> PresentStatus {
    auto semaphore = m_rhi.semaphore(signal_semaphore);
    auto result = acquire_next_image(semaphore, index);
    return map_present_result(result);
}

auto VulkanPresent::present_backbuffer(uint32_t index, RhiSemaphore wait_semaphore)
    -> PresentStatus {
    auto semaphore = m_rhi.semaphore(wait_semaphore);
    auto result = present(semaphore, index);
    return map_present_result(result);
}

void VulkanPresent::recreate_swapchain() {
    do_recreate_swapchain();
}

auto VulkanPresent::framebuffer_extent() const noexcept -> FramebufferExtent {
    return FramebufferExtent{m_extent.width, m_extent.height};
}

vk::Result VulkanPresent::acquire_next_image(vk::Semaphore semaphore, uint32_t* image_index) {
    try {
        return m_rhi.device().acquireNextImageKHR(m_swapchain.get(), UINT64_MAX, semaphore, {},
                                                  image_index);
    } catch (vk::OutOfDateKHRError const&) {
        return vk::Result::eErrorOutOfDateKHR;
    } catch (vk::SystemError const& err) {
        if (m_logger) {
            m_logger->error("Swapchain acquire failed: {}", err.what());
        }
        return static_cast<vk::Result>(err.code().value());
    }
}

vk::Result VulkanPresent::present(vk::Semaphore wait_semaphore, uint32_t image_index) {
    auto present_info = vk::PresentInfoKHR{}
                            .setWaitSemaphores(wait_semaphore)
                            .setSwapchains(m_swapchain.get())
                            .setImageIndices(image_index);
    try {
        return m_rhi.queue().presentKHR(present_info);
    } catch (vk::OutOfDateKHRError const&) {
        return vk::Result::eErrorOutOfDateKHR;
    } catch (vk::SystemError const& err) {
        if (m_logger) {
            m_logger->error("Swapchain present failed: {}", err.what());
        }
        return static_cast<vk::Result>(err.code().value());
    }
}

void VulkanPresent::resize_swapchain() {
    do_recreate_swapchain();
}

void VulkanPresent::create_surface() {
    auto handle = m_windowing.native_handle();
#if defined(_WIN32)
    if (!handle.platform_handle) {
        throw std::runtime_error("Windowing must provide a native platform handle");
    }
    auto const hwnd = static_cast<HWND>(handle.platform_handle);
    auto const hinstance = GetModuleHandle(nullptr);
    auto create_info = vk::Win32SurfaceCreateInfoKHR{}.setHinstance(hinstance).setHwnd(hwnd);
    auto created = m_rhi.instance().createWin32SurfaceKHR(create_info);
#elif defined(__linux__)
    if (!handle.platform_handle || !handle.window_handle) {
        throw std::runtime_error("Windowing must provide native display and window handles");
    }
    auto* display = static_cast<Display*>(handle.platform_handle);
    auto window = static_cast<Window>(reinterpret_cast<std::uintptr_t>(handle.window_handle));
    auto create_info = vk::XlibSurfaceCreateInfoKHR{}.setDpy(display).setWindow(window);
    auto created = m_rhi.instance().createXlibSurfaceKHR(create_info);
#else
    static_cast<void>(handle);
    throw std::runtime_error("Unsupported host platform for Vulkan present");
#endif
    m_logger->info("Vulkan surface created");
    m_surface = vk::UniqueSurfaceKHR{created, m_rhi.instance()};
}

void VulkanPresent::create_swapchain() {
    auto const caps = m_rhi.physical_device().getSurfaceCapabilitiesKHR(m_surface.get());
    auto const formats = m_rhi.physical_device().getSurfaceFormatsKHR(m_surface.get());
    auto const present_modes = m_rhi.physical_device().getSurfacePresentModesKHR(m_surface.get());

    // Some drivers reject SRGB swapchain formats when storage usage is implied.
    // Require formats that support the usage bits we rely on.
    auto const format_usage_check = vk::ImageUsageFlagBits::eColorAttachment |
                                    vk::ImageUsageFlagBits::eTransferSrc |
                                    vk::ImageUsageFlagBits::eStorage;

    auto format_supports_usage = [&](vk::Format format) {
        try {
            static_cast<void>(m_rhi.physical_device().getImageFormatProperties(
                format, vk::ImageType::e2D, vk::ImageTiling::eOptimal, format_usage_check, {}));
            return true;
        } catch (vk::SystemError const&) {
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
        auto const size = m_windowing.framebuffer_extent();
        extent.width = std::clamp(size.width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height =
            std::clamp(size.height, caps.minImageExtent.height, caps.maxImageExtent.height);
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
    swapchain_info.surface = m_surface.get();
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
    if (m_logger) {
        m_logger->info(
            "Creating swapchain: format={}, colorspace={}, present_mode={}, extent={}x{}, "
            "image_count={}",
            static_cast<int>(m_format), static_cast<int>(surface_format->colorSpace),
            static_cast<int>(present_mode), m_extent.width, m_extent.height, image_count);
    }
    m_swapchain = m_rhi.device().createSwapchainKHRUnique(swapchain_info);
    m_images = m_rhi.device().getSwapchainImagesKHR(m_swapchain.get());
}

void VulkanPresent::create_image_views() {
    m_image_views.clear();
    m_image_views.reserve(m_images.size());
    for (auto const& image : m_images) {
        auto view = m_rhi.device().createImageViewUnique(
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

void VulkanPresent::cleanup_swapchain() {
    if (!m_swapchain) {
        return;
    }
    m_rhi.device().waitIdle();
    if (m_logger) {
        m_logger->info("Swapchain resources destroyed");
    }
    m_image_views.clear();
    m_swapchain.reset();
    m_images.clear();
}

void VulkanPresent::do_recreate_swapchain() {
    if (m_logger) {
        m_logger->info("Recreating swapchain");
    }
    auto size = m_windowing.framebuffer_extent();
    while (size.width == 0 || size.height == 0) {
        m_windowing.wait_events();
        size = m_windowing.framebuffer_extent();
    }
    cleanup_swapchain();
    create_swapchain();
    create_image_views();
}
}  // namespace pts::rendering
