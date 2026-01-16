#pragma once

#include <core/legacy/application.h>
#include <core/legacy/callbackList.h>

#include <array>
#include <bitset>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

#include "debugDrawer.h"
#include "ext.h"
#include "inputAction.h"

namespace PTS {
/**
 * @brief abstract GLFW application. Responsible for creating the window and polling events.
 */
struct GLFWApplication : Application {
    // used to help detect if the mouse enters/leaves certain imgui windows
    struct ImGuiWindowInfo {
        CallbackList<void()> on_leave_region;
        CallbackList<void()> on_enter_region;
    };

    friend static void click_func(GLFWwindow* window, int button, int action, int mods);
    friend static void motion_func(GLFWwindow* window, double x, double y);
    friend static void scroll_func(GLFWwindow* window, double x, double y);
    friend static void key_func(GLFWwindow* window, int key, int scancode, int action, int mods);
    friend static void error_func(int error, const char* description);
    friend static void framebuffer_resize_func(GLFWwindow* window, int width, int height);

    NO_COPY_MOVE(GLFWApplication);

    GLFWApplication(std::string_view name, unsigned width, unsigned height, float min_frame_time);
    ~GLFWApplication() override;

    void run() override;

    [[nodiscard]] auto get_window_width() const noexcept -> int;
    [[nodiscard]] auto get_window_height() const noexcept -> int;
    /**
     * @brief Called every frame. Override to handle the main loop.
     * @param dt the time since the last frame
     */
    virtual void loop(float dt) = 0;

    // Vulkan/ImGui initialization (called after a renderer provides a Vulkan device)
    auto init_imgui_vulkan(vk::PhysicalDevice physical_device, vk::Device device,
                           uint32_t graphics_queue_family, vk::Queue graphics_queue) -> void;
    auto shutdown_imgui_vulkan() noexcept -> void;

    // Vulkan instance/surface accessors (instance is owned by GLFWApplication)
    [[nodiscard]] auto get_vk_instance() const noexcept -> vk::Instance {
        return m_vk_instance.get();
    }
    [[nodiscard]] auto get_vk_surface() const noexcept -> vk::SurfaceKHR {
        return m_vk_surface;
    }

   protected:
    virtual auto handle_input(InputEvent const& event) noexcept -> void {
    }
    virtual auto on_begin_first_loop() -> void;
    auto poll_input_events() noexcept -> void;

    /**
     * @brief Gets the renderer for the application.
     * @return the renderer
     */
    [[nodiscard]] auto get_debug_drawer() -> DebugDrawer& {
        return m_debug_drawer;
    }
    [[nodiscard]] auto get_cur_hovered_widget() const noexcept {
        return m_cur_hovered_widget;
    }
    [[nodiscard]] auto get_cur_focused_widget() const noexcept {
        return m_cur_focused_widget;
    }

    // imgui helpers
    auto get_imgui_window_info(std::string_view name) noexcept -> ImGuiWindowInfo& {
        // doesn't really care if the window exists or not
        return m_imgui_window_info[name];
    }

    auto begin_imgui_window(std::string_view name, ImGuiWindowFlags flags = 0) noexcept -> bool;

    void end_imgui_window() noexcept;
    auto get_window_content_pos(std::string_view name) const noexcept -> std::optional<ImVec2>;

    [[nodiscard]] auto get_time() const noexcept -> float override;
    [[nodiscard]] auto get_delta_time() const noexcept -> float override;

    auto set_min_frame_time(float min_frame_time) noexcept {
        m_min_frame_time = min_frame_time;
    }
    [[nodiscard]] auto get_min_frame_time() const noexcept {
        return m_min_frame_time;
    }

   protected:
    glm::vec2 m_mouse_scroll_delta;
    glm::vec2 m_mouse_pos;
    std::optional<glm::vec2> m_last_mouse_pos{std::nullopt};
    std::bitset<ImGuiMouseButton_COUNT> m_mouse_states{};
    std::bitset<ImGuiKey_COUNT> m_key_states{};
    std::array<std::string_view, ImGuiMouseButton_COUNT> m_mouse_initiated_window{};
    std::array<std::string_view, ImGuiKey_COUNT> m_key_initiated_window{};

    GLFWwindow* m_window;
    DebugDrawer m_debug_drawer;
    float m_min_frame_time;
    float m_delta_time{0.0f};
    std::unordered_map<std::string_view, ImGuiWindowInfo> m_imgui_window_info;

    std::string_view m_cur_hovered_widget, m_prev_hovered_widget;
    std::string_view m_cur_focused_widget;

    static constexpr auto k_no_hovered_widget = "";

    // Vulkan rendering for ImGui
    auto create_vulkan_instance() -> void;
    auto create_vulkan_surface() -> void;
    auto create_swapchain() -> void;
    auto create_render_pass() -> void;
    auto create_framebuffers() -> void;
    auto create_command_pool() -> void;
    auto create_command_buffers() -> void;
    auto create_sync_objects() -> void;
    auto cleanup_swapchain() -> void;
    auto recreate_swapchain() -> void;
    auto record_command_buffer(vk::CommandBuffer cmd_buf, uint32_t image_index) -> void;
    auto render_frame() -> void;

    vk::UniqueInstance m_vk_instance;
    vk::SurfaceKHR m_vk_surface{VK_NULL_HANDLE};
    vk::PhysicalDevice m_vk_physical_device{};
    vk::Device m_vk_device{};
    vk::Queue m_vk_graphics_queue{};
    uint32_t m_vk_graphics_queue_family{0};
    vk::UniqueSwapchainKHR m_vk_swapchain;
    vk::Format m_vk_swapchain_format{vk::Format::eUndefined};
    vk::Extent2D m_vk_swapchain_extent{};
    std::vector<vk::Image> m_vk_swapchain_images;
    std::vector<vk::UniqueImageView> m_vk_swapchain_image_views;
    vk::UniqueRenderPass m_vk_render_pass;
    std::vector<vk::UniqueFramebuffer> m_vk_framebuffers;
    vk::UniqueCommandPool m_vk_command_pool;
    std::vector<vk::UniqueCommandBuffer> m_vk_command_buffers;
    std::vector<vk::UniqueSemaphore> m_vk_image_available_semaphores;
    std::vector<vk::UniqueSemaphore> m_vk_render_finished_semaphores;
    std::vector<vk::UniqueFence> m_vk_in_flight_fences;
    size_t m_vk_frame_index{0};
    bool m_vk_framebuffer_resized{false};
    bool m_imgui_vulkan_initialized{false};
    vk::UniqueDescriptorPool m_imgui_descriptor_pool;
};
}  // namespace PTS
