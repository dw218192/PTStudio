#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <core/rendering/imguiVulkanPresenter.h>
#include <core/rendering/renderGraphHost.h>
#include <core/rendering/renderingHost.h>
#include <core/rendering/swapchainHost.h>
#include <core/rendering/vulkanContext.h>

#include <cstdlib>
#include <iostream>
#include <vector>

namespace PTS::rendering {
struct RenderingHost::Impl {
    explicit Impl(GLFWwindow* window) : window(window) {
        create_vulkan_instance();
        create_surface();
        context = std::make_unique<VulkanContext>(instance.get(), surface);
        swapchain = std::make_unique<SwapchainHost>(window, *context);
        render_graph =
            std::make_unique<RenderGraphHost>(context->physical_device(), context->device(),
                                              context->queue(), context->queue_family());
        presenter = std::make_unique<ImGuiVulkanPresenter>(window, *swapchain, *context);
        resize_render_graph(swapchain->extent().width, swapchain->extent().height);
    }

    ~Impl() {
        if (output_id) {
            presenter->unregister_texture(output_id);
            output_id = nullptr;
        }
        presenter.reset();
        render_graph.reset();
        swapchain.reset();
        context.reset();
        if (surface != VK_NULL_HANDLE) {
            instance->destroySurfaceKHR(surface);
            surface = VK_NULL_HANDLE;
        }
    }

    void resize_render_graph(uint32_t width, uint32_t height) {
        if (output_id) {
            presenter->unregister_texture(output_id);
            output_id = nullptr;
        }
        render_graph->resize(width, height);
        if (width == 0 || height == 0) {
            return;
        }
        output_id = presenter->register_texture(render_graph->output_sampler(),
                                                render_graph->output_image_view(),
                                                render_graph->output_layout());
    }

    void create_vulkan_instance() {
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
        uint32_t glfw_ext_count = 0;
        auto glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
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

        instance = vk::createInstanceUnique(
            vk::InstanceCreateInfo{instance_flags, &app_info, layers, extensions});
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());
    }

    void create_surface() {
        VkSurfaceKHR created = VK_NULL_HANDLE;
        if (glfwCreateWindowSurface(instance.get(), window, nullptr, &created) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan surface" << std::endl;
            std::exit(-1);
        }
        surface = created;
    }

    GLFWwindow* window{nullptr};
    vk::UniqueInstance instance;
    vk::SurfaceKHR surface{VK_NULL_HANDLE};
    std::unique_ptr<VulkanContext> context;
    std::unique_ptr<SwapchainHost> swapchain;
    std::unique_ptr<RenderGraphHost> render_graph;
    std::unique_ptr<ImGuiVulkanPresenter> presenter;
    ImTextureID output_id{nullptr};
};

RenderingHost::RenderingHost(GLFWwindow* window) : m_impl(std::make_unique<Impl>(window)) {
}

RenderingHost::~RenderingHost() = default;

void RenderingHost::new_frame() {
    m_impl->presenter->new_frame();
}

void RenderingHost::render(bool framebuffer_resized) {
    m_impl->presenter->render(framebuffer_resized);
}

void RenderingHost::resize_render_graph(uint32_t width, uint32_t height) {
    m_impl->resize_render_graph(width, height);
}

void RenderingHost::set_render_graph_current() {
    if (m_impl->render_graph) {
        m_impl->render_graph->set_current();
    }
}

void RenderingHost::clear_render_graph_current() {
    if (m_impl->render_graph) {
        m_impl->render_graph->clear_current();
    }
}

auto RenderingHost::render_graph_api() const noexcept -> const PtsRenderGraphApi* {
    return m_impl->render_graph ? m_impl->render_graph->api() : nullptr;
}

auto RenderingHost::output_texture() const noexcept -> PtsTexture {
    return m_impl->render_graph ? m_impl->render_graph->output_texture() : PtsTexture{};
}

auto RenderingHost::output_imgui_id() const noexcept -> ImTextureID {
    return m_impl->output_id;
}
}  // namespace PTS::rendering
