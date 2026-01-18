#include "vulkanBackend.h"

#include <cstdlib>
#include <vector>

#include "imguiVulkanPresenter.h"
#include "renderGraphHost.h"
#include "swapchainHost.h"
#include "vulkanContext.h"

namespace pts::rendering {

VulkanBackend::VulkanBackend(GLFWwindow* window, LoggingManager& logging_manager)
    : m_window(window), m_logging_manager(logging_manager) {
    m_logger = logging_manager.get_logger_shared("VulkanBackend");
    m_logger->info("Vulkan backend initializing");
    create_vulkan_instance();
    create_surface();
    m_context = std::make_unique<VulkanContext>(m_instance.get(), m_surface, m_logging_manager);
    m_swapchain = std::make_unique<SwapchainHost>(m_window, *m_context, m_logging_manager);
    m_render_graph = std::make_unique<RenderGraphHost>(
        m_context->physical_device(), m_context->device(), m_context->queue(),
        m_context->queue_family(), m_logging_manager);
    m_presenter = std::make_unique<ImGuiVulkanPresenter>(m_window, *m_swapchain, *m_context,
                                                         m_logging_manager);
    resize_render_graph(m_swapchain->extent().width, m_swapchain->extent().height);
    m_logger->info("Vulkan backend initialized");
}

VulkanBackend::~VulkanBackend() {
    m_logger->info("Vulkan backend shutting down");
    if (m_output_id) {
        m_context->device().waitIdle();
        m_presenter->unregister_texture(m_output_id);
        m_output_id = nullptr;
    }
    m_presenter.reset();
    m_render_graph.reset();
    m_swapchain.reset();
    m_context.reset();
    if (m_surface != VK_NULL_HANDLE) {
        m_instance->destroySurfaceKHR(m_surface);
        m_surface = VK_NULL_HANDLE;
    }
}

void VulkanBackend::new_frame() {
    m_presenter->new_frame();
}

void VulkanBackend::render(bool framebuffer_resized) {
    m_presenter->render(framebuffer_resized);
}

void VulkanBackend::resize_render_graph(uint32_t width, uint32_t height) {
    if (m_output_id) {
        m_context->device().waitIdle();
        m_presenter->unregister_texture(m_output_id);
        m_output_id = nullptr;
    }
    m_render_graph->resize(width, height);
    if (width == 0 || height == 0) {
        m_logger->debug("Render graph resize skipped for zero extent");
        return;
    }
    m_output_id = m_presenter->register_texture(m_render_graph->output_sampler(),
                                                m_render_graph->output_image_view(),
                                                m_render_graph->output_layout());
}

void VulkanBackend::set_render_graph_current() {
    if (m_render_graph) {
        m_render_graph->set_current();
    }
}

void VulkanBackend::clear_render_graph_current() {
    if (m_render_graph) {
        m_render_graph->clear_current();
    }
}

auto VulkanBackend::render_graph_api() const noexcept -> const PtsRenderGraphApi* {
    return m_render_graph ? m_render_graph->api() : nullptr;
}

auto VulkanBackend::output_texture() const noexcept -> PtsTexture {
    return m_render_graph ? m_render_graph->output_texture() : PtsTexture{};
}

auto VulkanBackend::output_imgui_id() const noexcept -> ImTextureID {
    return m_output_id;
}

void VulkanBackend::create_vulkan_instance() {
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
    uint32_t glfw_ext_count = 0;
    auto glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
    if (!glfw_exts) {
        throw std::runtime_error("Failed to get required Vulkan instance extensions from GLFW");
    }
    auto extensions = std::vector<char const*>{glfw_exts, glfw_exts + glfw_ext_count};
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    auto instance_flags = vk::InstanceCreateFlags{};
#ifdef VK_KHR_portability_enumeration
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instance_flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

    auto layers = std::vector<char const*>{};
#ifndef NDEBUG
    layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    auto const app_info = vk::ApplicationInfo{"PTS Editor", VK_MAKE_VERSION(1, 0, 0), "PTS",
                                              VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_2};

    m_instance = vk::createInstanceUnique(
        vk::InstanceCreateInfo{instance_flags, &app_info, layers, extensions});
    VULKAN_HPP_DEFAULT_DISPATCHER.init(m_instance.get());
    m_logger->info("Vulkan instance created (extensions={}, layers={})", extensions.size(),
                   layers.size());
}

void VulkanBackend::create_surface() {
    VkSurfaceKHR created = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(m_instance.get(), m_window, nullptr, &created) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan surface");
    }
    m_surface = created;
    m_logger->info("Vulkan surface created");
}

}  // namespace pts::rendering
