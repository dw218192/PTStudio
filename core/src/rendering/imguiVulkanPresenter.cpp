#include <GLFW/glfw3.h>
#include <core/rendering/imguiVulkanPresenter.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <array>
#include <cstdlib>
#include <iostream>

namespace PTS::rendering {
ImGuiVulkanPresenter::ImGuiVulkanPresenter(GLFWwindow* window, SwapchainHost& swapchain,
                                           VulkanContext& context)
    : m_window(window), m_swapchain(swapchain), m_context(context) {
    create_command_pool();
    create_render_pass();
    create_framebuffers();
    create_command_buffers();
    create_sync_objects();
    init_imgui_backend();
}

ImGuiVulkanPresenter::~ImGuiVulkanPresenter() {
    if (m_initialized) {
        m_context.device().waitIdle();
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
    }
    cleanup_swapchain();
    if (m_command_pool) {
        m_context.device().destroyCommandPool(m_command_pool);
    }
}

void ImGuiVulkanPresenter::new_frame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
}

void ImGuiVulkanPresenter::render(bool framebuffer_resized) {
    if (!m_initialized) {
        return;
    }

    static constexpr uint32_t k_frames_in_flight = 2;
    auto const frame_index = m_frame_index_counter % k_frames_in_flight;
    (void) m_context.device().waitForFences(m_in_flight_fences[frame_index].get(), true,
                                            UINT64_MAX);

    uint32_t image_index = 0;
    auto acquire_result = m_swapchain.acquire_next_image(
        m_image_available_semaphores[frame_index].get(), &image_index);
    if (acquire_result == vk::Result::eErrorOutOfDateKHR ||
        acquire_result == vk::Result::eSuboptimalKHR) {
        recreate_swapchain();
        return;
    }

    m_context.device().resetFences(m_in_flight_fences[frame_index].get());
    m_command_buffers[frame_index]->reset();
    record_command_buffer(m_command_buffers[frame_index].get(), image_index);

    auto const wait_stages =
        std::array{vk::PipelineStageFlags{vk::PipelineStageFlagBits::eColorAttachmentOutput}};
    auto submit_info = vk::SubmitInfo{}
                           .setWaitSemaphores(m_image_available_semaphores[frame_index].get())
                           .setWaitDstStageMask(wait_stages)
                           .setCommandBuffers(m_command_buffers[frame_index].get())
                           .setSignalSemaphores(m_render_finished_semaphores[frame_index].get());

    m_context.queue().submit(submit_info, m_in_flight_fences[frame_index].get());

    auto present_result =
        m_swapchain.present(m_render_finished_semaphores[frame_index].get(), image_index);

    if (present_result == vk::Result::eErrorOutOfDateKHR ||
        present_result == vk::Result::eSuboptimalKHR || framebuffer_resized) {
        recreate_swapchain();
    }

    m_frame_index_counter = (m_frame_index_counter + 1) % k_frames_in_flight;
}

void ImGuiVulkanPresenter::resize() {
    recreate_swapchain();
}

ImTextureID ImGuiVulkanPresenter::register_texture(vk::Sampler sampler, vk::ImageView view,
                                                   vk::ImageLayout layout) {
    if (!m_initialized) {
        return nullptr;
    }
    return ImGui_ImplVulkan_AddTexture(sampler, view, static_cast<VkImageLayout>(layout));
}

void ImGuiVulkanPresenter::unregister_texture(ImTextureID id) {
    if (id) {
        ImGui_ImplVulkan_RemoveTexture(static_cast<VkDescriptorSet>(id));
    }
}

void ImGuiVulkanPresenter::create_render_pass() {
    auto color_attachment = vk::AttachmentDescription{}
                                .setFormat(m_swapchain.format())
                                .setSamples(vk::SampleCountFlagBits::e1)
                                .setLoadOp(vk::AttachmentLoadOp::eClear)
                                .setStoreOp(vk::AttachmentStoreOp::eStore)
                                .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                                .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                                .setInitialLayout(vk::ImageLayout::eUndefined)
                                .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    auto color_ref = vk::AttachmentReference{}.setAttachment(0).setLayout(
        vk::ImageLayout::eColorAttachmentOptimal);

    auto subpass = vk::SubpassDescription{}
                       .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                       .setColorAttachments(color_ref);

    auto dependency = vk::SubpassDependency{}
                          .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                          .setDstSubpass(0)
                          .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                          .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                          .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

    m_render_pass = m_context.device().createRenderPassUnique(vk::RenderPassCreateInfo{}
                                                                  .setAttachments(color_attachment)
                                                                  .setSubpasses(subpass)
                                                                  .setDependencies(dependency));
}

void ImGuiVulkanPresenter::create_framebuffers() {
    m_framebuffers.clear();
    m_framebuffers.reserve(m_swapchain.image_views().size());
    for (auto const& view : m_swapchain.image_views()) {
        auto framebuffer =
            m_context.device().createFramebufferUnique(vk::FramebufferCreateInfo{}
                                                           .setRenderPass(m_render_pass.get())
                                                           .setAttachments(view.get())
                                                           .setWidth(m_swapchain.extent().width)
                                                           .setHeight(m_swapchain.extent().height)
                                                           .setLayers(1));
        m_framebuffers.emplace_back(std::move(framebuffer));
    }
}

void ImGuiVulkanPresenter::create_command_pool() {
    m_command_pool = m_context.device().createCommandPool(
        vk::CommandPoolCreateInfo{}
            .setQueueFamilyIndex(m_context.queue_family())
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer));
}

void ImGuiVulkanPresenter::create_command_buffers() {
    static constexpr uint32_t k_frames_in_flight = 2;
    m_command_buffers =
        m_context.device().allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{
            m_command_pool, vk::CommandBufferLevel::ePrimary, k_frames_in_flight});
}

void ImGuiVulkanPresenter::create_sync_objects() {
    static constexpr uint32_t k_frames_in_flight = 2;
    m_image_available_semaphores.resize(k_frames_in_flight);
    m_render_finished_semaphores.resize(k_frames_in_flight);
    m_in_flight_fences.resize(k_frames_in_flight);
    for (uint32_t i = 0; i < k_frames_in_flight; ++i) {
        m_image_available_semaphores[i] = m_context.device().createSemaphoreUnique({});
        m_render_finished_semaphores[i] = m_context.device().createSemaphoreUnique({});
        m_in_flight_fences[i] = m_context.device().createFenceUnique(
            vk::FenceCreateInfo{}.setFlags(vk::FenceCreateFlagBits::eSignaled));
    }
}

void ImGuiVulkanPresenter::cleanup_swapchain() {
    m_framebuffers.clear();
    m_render_pass.reset();
}

void ImGuiVulkanPresenter::recreate_swapchain() {
    cleanup_swapchain();
    m_swapchain.resize();
    create_render_pass();
    create_framebuffers();
    ImGui_ImplVulkan_SetMinImageCount(m_swapchain.image_count());
}

void ImGuiVulkanPresenter::record_command_buffer(vk::CommandBuffer cmd_buf, uint32_t image_index) {
    cmd_buf.begin(vk::CommandBufferBeginInfo{});
    auto clear_color = vk::ClearColorValue(std::array<float, 4>{0, 0, 0, 1});
    auto clear_value = vk::ClearValue(clear_color);
    auto render_pass_info = vk::RenderPassBeginInfo{}
                                .setRenderPass(m_render_pass.get())
                                .setFramebuffer(m_framebuffers[image_index].get())
                                .setRenderArea(vk::Rect2D{{0, 0}, m_swapchain.extent()})
                                .setClearValues(clear_value);

    cmd_buf.beginRenderPass(render_pass_info, vk::SubpassContents::eInline);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd_buf);
    cmd_buf.endRenderPass();
    cmd_buf.end();
}

void ImGuiVulkanPresenter::init_imgui_backend() {
    auto const pool_sizes = std::array{
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageTexelBuffer, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBufferDynamic, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBufferDynamic, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eInputAttachment, 1000},
    };

    m_imgui_descriptor_pool = m_context.device().createDescriptorPoolUnique(
        vk::DescriptorPoolCreateInfo{}
            .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
            .setMaxSets(1000 * static_cast<uint32_t>(pool_sizes.size()))
            .setPoolSizes(pool_sizes));

    ImGui_ImplGlfw_InitForVulkan(m_window, true);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = m_context.instance();
    init_info.PhysicalDevice = m_context.physical_device();
    init_info.Device = m_context.device();
    init_info.QueueFamily = m_context.queue_family();
    init_info.Queue = m_context.queue();
    init_info.DescriptorPool = m_imgui_descriptor_pool.get();
    init_info.MinImageCount = m_swapchain.image_count();
    init_info.ImageCount = m_swapchain.image_count();
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.CheckVkResultFn = [](VkResult err) {
        if (err != VK_SUCCESS) {
            std::cerr << "ImGui Vulkan error: " << err << std::endl;
        }
    };

    ImGui_ImplVulkan_Init(&init_info, m_render_pass.get());
    if (!ImGui_ImplVulkan_CreateFontsTexture()) {
        std::cerr << "Failed to upload ImGui Vulkan fonts" << std::endl;
        std::exit(-1);
    }
    m_initialized = true;
}
}  // namespace PTS::rendering
