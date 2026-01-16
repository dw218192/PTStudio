#pragma once

#include <core/rendering/swapchainHost.h>
#include <imgui.h>

struct GLFWwindow;

namespace pts::rendering {
class ImGuiVulkanPresenter {
   public:
    ImGuiVulkanPresenter(GLFWwindow* window, SwapchainHost& swapchain, VulkanContext& context);
    ~ImGuiVulkanPresenter();

    ImGuiVulkanPresenter(const ImGuiVulkanPresenter&) = delete;
    ImGuiVulkanPresenter& operator=(const ImGuiVulkanPresenter&) = delete;
    ImGuiVulkanPresenter(ImGuiVulkanPresenter&&) = delete;
    ImGuiVulkanPresenter& operator=(ImGuiVulkanPresenter&&) = delete;

    void new_frame();
    void render(bool framebuffer_resized);
    void resize();

    [[nodiscard]] auto register_texture(vk::Sampler sampler, vk::ImageView view,
                                        vk::ImageLayout layout) -> ImTextureID;
    void unregister_texture(ImTextureID id);

   private:
    void create_swapchain();
    void create_render_pass();
    void create_framebuffers();
    void create_command_pool();
    void create_command_buffers();
    void create_sync_objects();
    void cleanup_swapchain();
    void recreate_swapchain();
    void record_command_buffer(vk::CommandBuffer cmd_buf, uint32_t image_index);
    void init_imgui_backend();

    GLFWwindow* m_window{nullptr};
    SwapchainHost& m_swapchain;
    VulkanContext& m_context;

    vk::UniqueRenderPass m_render_pass;
    std::vector<vk::UniqueFramebuffer> m_framebuffers;
    vk::CommandPool m_command_pool{};
    std::vector<vk::UniqueCommandBuffer> m_command_buffers;
    std::vector<vk::UniqueSemaphore> m_image_available_semaphores;
    std::vector<vk::UniqueSemaphore> m_render_finished_semaphores;
    std::vector<vk::UniqueFence> m_in_flight_fences;
    size_t m_frame_index_counter{0};
    bool m_initialized{false};
    vk::UniqueDescriptorPool m_imgui_descriptor_pool;
};
}  // namespace pts::rendering
