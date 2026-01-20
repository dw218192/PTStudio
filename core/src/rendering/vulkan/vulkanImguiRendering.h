#pragma once

#include <core/loggingManager.h>

#include <memory>
#include <vector>

#include "../imguiBackend.h"
#include "vulkanPresent.h"
#include "vulkanRhi.h"

namespace pts::rendering {
class VulkanImguiRendering final : public IImguiRendering {
   public:
    VulkanImguiRendering(std::shared_ptr<VulkanRhi> rhi, std::shared_ptr<VulkanPresent> present,
                         pts::LoggingManager& logging_manager);
    ~VulkanImguiRendering() override;

    VulkanImguiRendering(const VulkanImguiRendering&) = delete;
    VulkanImguiRendering& operator=(const VulkanImguiRendering&) = delete;
    VulkanImguiRendering(VulkanImguiRendering&&) = delete;
    VulkanImguiRendering& operator=(VulkanImguiRendering&&) = delete;

    void new_frame() override;
    void render(bool framebuffer_resized) override;
    void resize() override;
    auto set_render_output(IRenderGraph& render_graph) -> ImTextureID override;
    void clear_render_output() override;
    [[nodiscard]] auto output_id() const noexcept -> ImTextureID override {
        return m_output_id;
    }

   private:
    void create_render_pass();
    void create_framebuffers();
    void create_command_pool();
    void create_command_buffers();
    void create_sync_objects();
    void cleanup_swapchain();
    void recreate_swapchain();
    void record_command_buffer(vk::CommandBuffer cmd_buf, uint32_t image_index);
    void init_imgui_backend();

    std::shared_ptr<VulkanRhi> m_rhi_owner;
    std::shared_ptr<VulkanPresent> m_present_owner;
    VulkanPresent& m_present;
    VulkanRhi& m_rhi;

    vk::UniqueRenderPass m_render_pass;
    std::vector<vk::UniqueFramebuffer> m_framebuffers;
    vk::UniqueCommandPool m_command_pool{};
    std::vector<vk::UniqueCommandBuffer> m_command_buffers;
    std::vector<vk::UniqueSemaphore> m_image_available_semaphores;
    std::vector<vk::UniqueSemaphore> m_render_finished_semaphores;
    std::vector<vk::UniqueFence> m_in_flight_fences;
    size_t m_frame_index_counter{0};
    bool m_initialized{false};
    vk::UniqueDescriptorPool m_imgui_descriptor_pool;
    ImTextureID m_output_id{ImTextureID_Invalid};
    std::shared_ptr<spdlog::logger> m_logger;
};
}  // namespace pts::rendering
