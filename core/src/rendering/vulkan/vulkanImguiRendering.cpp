#include "vulkanImguiRendering.h"

#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <spdlog/spdlog.h>

#include <array>
#include <stdexcept>
#include <utility>

#include "vulkanRenderGraph.h"

namespace {
std::weak_ptr<spdlog::logger> g_imgui_logger;

void log_imgui_vk_result(VkResult err) {
    if (err == VK_SUCCESS) {
        return;
    }
    if (auto logger = g_imgui_logger.lock()) {
        logger->error("ImGui Vulkan error: {}", static_cast<int>(err));
    }
}
}  // namespace

namespace pts::rendering {
VulkanImguiRendering::VulkanImguiRendering(std::shared_ptr<VulkanRhi> rhi,
                                           std::shared_ptr<VulkanPresent> present,
                                           pts::LoggingManager& logging_manager)
    : m_rhi_owner(std::move(rhi)),
      m_present_owner(std::move(present)),
      m_present(*m_present_owner),
      m_rhi(*m_rhi_owner) {
    m_logger = logging_manager.get_logger_shared("ImGuiRendering");
    g_imgui_logger = m_logger;
    create_command_pool();
    create_render_pass();
    create_framebuffers();
    create_command_buffers();
    create_sync_objects();
    init_imgui_backend();
    m_logger->info("ImGui Vulkan rendering initialized");
}

VulkanImguiRendering::~VulkanImguiRendering() {
    if (m_initialized) {
        m_rhi.device().waitIdle();
        clear_render_output();
        ImGui_ImplVulkan_Shutdown();
    }
    cleanup_swapchain();
    if (m_logger) {
        m_logger->info("ImGui Vulkan rendering destroyed");
    }
}

void VulkanImguiRendering::new_frame() {
    ImGui_ImplVulkan_NewFrame();
}

void VulkanImguiRendering::render(bool framebuffer_resized) {
    if (!m_initialized) {
        return;
    }

    static constexpr uint32_t k_frames_in_flight = 2;
    auto const frame_index = m_frame_index_counter % k_frames_in_flight;
    (void) m_rhi.device().waitForFences(m_in_flight_fences[frame_index].get(), true, UINT64_MAX);

    uint32_t image_index = 0;
    auto acquire_result =
        m_present.acquire_next_image(m_image_available_semaphores[frame_index].get(), &image_index);
    if (acquire_result == vk::Result::eErrorOutOfDateKHR ||
        acquire_result == vk::Result::eSuboptimalKHR) {
        m_logger->info("Swapchain out of date; recreating");
        recreate_swapchain();
        return;
    }

    m_rhi.device().resetFences(m_in_flight_fences[frame_index].get());
    m_command_buffers[frame_index]->reset();
    record_command_buffer(m_command_buffers[frame_index].get(), image_index);

    auto const wait_stages =
        std::array{vk::PipelineStageFlags{vk::PipelineStageFlagBits::eColorAttachmentOutput}};
    auto submit_info = vk::SubmitInfo{}
                           .setWaitSemaphores(m_image_available_semaphores[frame_index].get())
                           .setWaitDstStageMask(wait_stages)
                           .setCommandBuffers(m_command_buffers[frame_index].get())
                           .setSignalSemaphores(m_render_finished_semaphores[frame_index].get());

    m_rhi.queue().submit(submit_info, m_in_flight_fences[frame_index].get());

    auto present_result =
        m_present.present(m_render_finished_semaphores[frame_index].get(), image_index);

    if (present_result == vk::Result::eErrorOutOfDateKHR ||
        present_result == vk::Result::eSuboptimalKHR || framebuffer_resized) {
        m_logger->info("Swapchain out of date after present; recreating");
        recreate_swapchain();
    }

    m_frame_index_counter = (m_frame_index_counter + 1) % k_frames_in_flight;
}

void VulkanImguiRendering::resize() {
    recreate_swapchain();
}

auto VulkanImguiRendering::set_render_output(IRenderGraph& render_graph) -> ImTextureID {
    auto* vulkan_graph = dynamic_cast<VulkanRenderGraph*>(&render_graph);
    if (!vulkan_graph) {
        if (m_logger) {
            m_logger->error("ImGui rendering requires Vulkan render graph output");
        }
        return ImTextureID_Invalid;
    }
    clear_render_output();
    if (!m_initialized) {
        return ImTextureID_Invalid;
    }
    auto descriptor = ImGui_ImplVulkan_AddTexture(
        vulkan_graph->output_sampler(), vulkan_graph->output_image_view(),
        static_cast<VkImageLayout>(vulkan_graph->output_layout()));
    m_output_id = static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(descriptor));
    return m_output_id;
}

void VulkanImguiRendering::clear_render_output() {
    if (m_output_id != ImTextureID_Invalid) {
        auto descriptor = reinterpret_cast<VkDescriptorSet>(static_cast<uintptr_t>(m_output_id));
        ImGui_ImplVulkan_RemoveTexture(descriptor);
        m_output_id = ImTextureID_Invalid;
    }
}

void VulkanImguiRendering::create_render_pass() {
    auto color_attachment = vk::AttachmentDescription{}
                                .setFormat(m_present.format())
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

    m_render_pass = m_rhi.device().createRenderPassUnique(vk::RenderPassCreateInfo{}
                                                              .setAttachments(color_attachment)
                                                              .setSubpasses(subpass)
                                                              .setDependencies(dependency));
}

void VulkanImguiRendering::create_framebuffers() {
    m_framebuffers.clear();
    m_framebuffers.reserve(m_present.image_views().size());
    for (auto const& view : m_present.image_views()) {
        auto framebuffer =
            m_rhi.device().createFramebufferUnique(vk::FramebufferCreateInfo{}
                                                       .setRenderPass(m_render_pass.get())
                                                       .setAttachments(view.get())
                                                       .setWidth(m_present.extent().width)
                                                       .setHeight(m_present.extent().height)
                                                       .setLayers(1));
        m_framebuffers.emplace_back(std::move(framebuffer));
    }
}

void VulkanImguiRendering::create_command_pool() {
    m_command_pool = m_rhi.device().createCommandPoolUnique(
        vk::CommandPoolCreateInfo{}
            .setQueueFamilyIndex(m_rhi.queue_family())
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer));
}

void VulkanImguiRendering::create_command_buffers() {
    static constexpr uint32_t k_frames_in_flight = 2;
    m_command_buffers = m_rhi.device().allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{
        m_command_pool.get(), vk::CommandBufferLevel::ePrimary, k_frames_in_flight});
}

void VulkanImguiRendering::create_sync_objects() {
    static constexpr uint32_t k_frames_in_flight = 2;
    m_image_available_semaphores.resize(k_frames_in_flight);
    m_render_finished_semaphores.resize(k_frames_in_flight);
    m_in_flight_fences.resize(k_frames_in_flight);
    for (uint32_t i = 0; i < k_frames_in_flight; ++i) {
        m_image_available_semaphores[i] = m_rhi.device().createSemaphoreUnique({});
        m_render_finished_semaphores[i] = m_rhi.device().createSemaphoreUnique({});
        m_in_flight_fences[i] = m_rhi.device().createFenceUnique(
            vk::FenceCreateInfo{}.setFlags(vk::FenceCreateFlagBits::eSignaled));
    }
}

void VulkanImguiRendering::cleanup_swapchain() {
    if (m_render_pass || !m_framebuffers.empty()) {
        m_rhi.device().waitIdle();
    }
    m_framebuffers.clear();
    m_render_pass.reset();
}

void VulkanImguiRendering::recreate_swapchain() {
    m_logger->info("Recreating ImGui swapchain resources");
    cleanup_swapchain();
    m_present.resize_swapchain();
    create_render_pass();
    create_framebuffers();
    ImGui_ImplVulkan_SetMinImageCount(m_present.image_count());
}

void VulkanImguiRendering::record_command_buffer(vk::CommandBuffer cmd_buf, uint32_t image_index) {
    cmd_buf.begin(vk::CommandBufferBeginInfo{});
    auto clear_color = vk::ClearColorValue(std::array<float, 4>{0, 0, 0, 1});
    auto clear_value = vk::ClearValue(clear_color);
    auto render_pass_info = vk::RenderPassBeginInfo{}
                                .setRenderPass(m_render_pass.get())
                                .setFramebuffer(m_framebuffers[image_index].get())
                                .setRenderArea(vk::Rect2D{{0, 0}, m_present.extent()})
                                .setClearValues(clear_value);

    cmd_buf.beginRenderPass(render_pass_info, vk::SubpassContents::eInline);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd_buf);
    cmd_buf.endRenderPass();
    cmd_buf.end();
}

void VulkanImguiRendering::init_imgui_backend() {
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

    m_imgui_descriptor_pool = m_rhi.device().createDescriptorPoolUnique(
        vk::DescriptorPoolCreateInfo{}
            .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
            .setMaxSets(1000 * static_cast<uint32_t>(pool_sizes.size()))
            .setPoolSizes(pool_sizes));

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = m_rhi.instance();
    init_info.PhysicalDevice = m_rhi.physical_device();
    init_info.Device = m_rhi.device();
    init_info.QueueFamily = m_rhi.queue_family();
    init_info.Queue = m_rhi.queue();
    init_info.DescriptorPool = m_imgui_descriptor_pool.get();
    init_info.MinImageCount = m_present.image_count();
    init_info.ImageCount = m_present.image_count();
    init_info.PipelineInfoMain.RenderPass = m_render_pass.get();
    init_info.PipelineInfoMain.Subpass = 0;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.CheckVkResultFn = log_imgui_vk_result;

    ImGui_ImplVulkan_Init(&init_info);
    m_initialized = true;
}
}  // namespace pts::rendering
