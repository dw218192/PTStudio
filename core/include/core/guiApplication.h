#pragma once

#include <core/application.h>
#include <core/inputAction.h>
#include <core/rendering/windowing.h>
#include <core/signal.h>
#include <imgui.h>

#include <array>
#include <bitset>
#include <glm/glm.hpp>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace pts {
namespace rendering {
class WebGpuContext;
class IImguiRendering;
class IImguiWindowing;
}  // namespace rendering

/**
 * @brief GUI application with windowing, WebGPU rendering, and ImGui support.
 *
 * Extends Application with a window, a WebGPU context bound to that window,
 * and an ImGui rendering pipeline.
 *
 * Subclasses override loop() to issue ImGui draw calls and/or custom
 * WebGPU rendering. Override run_one_frame() to bypass the ImGui frame.
 */
struct GUIApplication : Application {
    // used to help detect if the mouse enters/leaves certain imgui windows
    struct ImGuiWindowInfo {
        Signal<void()> on_leave_region;
        Signal<void()> on_enter_region;
    };

    NO_COPY_MOVE(GUIApplication);

    GUIApplication(std::string_view name, pts::LoggingManager& logging_manager);
    ~GUIApplication() override;

    void run() override;

    void on_scroll_event(double x, double y) noexcept;

    [[nodiscard]] auto get_window_width() const noexcept -> int;
    [[nodiscard]] auto get_window_height() const noexcept -> int;

   protected:
    virtual auto handle_input(InputEvent const& event) noexcept -> void {
    }
    virtual auto on_begin_first_loop() -> void;
    auto poll_input_events() noexcept -> void;
    [[nodiscard]] auto get_window_extent() const noexcept -> glm::ivec2;
    auto set_cursor_pos(float x, float y) noexcept -> void;

    [[nodiscard]] auto get_cur_hovered_widget() const noexcept {
        return m_cur_hovered_widget;
    }
    [[nodiscard]] auto get_cur_focused_widget() const noexcept {
        return m_cur_focused_widget;
    }

    // imgui helpers
    auto get_imgui_window_info(std::string_view name) noexcept -> ImGuiWindowInfo& {
        return m_imgui_window_info[name];
    }

    auto begin_imgui_window(std::string_view name, ImGuiWindowFlags flags = 0) noexcept -> bool;

    void end_imgui_window() noexcept;
    auto get_window_content_pos(std::string_view name) const noexcept -> std::optional<ImVec2>;

    // Windowing / rendering accessors for subclasses
    [[nodiscard]] auto get_webgpu_context() noexcept -> pts::rendering::WebGpuContext* {
        return m_webgpu_context.get();
    }
    [[nodiscard]] auto get_webgpu_context() const noexcept -> const pts::rendering::WebGpuContext* {
        return m_webgpu_context.get();
    }

    [[nodiscard]] auto get_windowing() noexcept -> pts::rendering::IWindowing* {
        return m_windowing.get();
    }
    [[nodiscard]] auto get_windowing() const noexcept -> const pts::rendering::IWindowing* {
        return m_windowing.get();
    }

    [[nodiscard]] auto get_viewport() noexcept -> pts::rendering::IViewport* {
        return m_viewport.get();
    }
    [[nodiscard]] auto get_viewport() const noexcept -> const pts::rendering::IViewport* {
        return m_viewport.get();
    }

    void on_framebuffer_resized() noexcept;

    /**
     * @brief Drive WebGPU async init forward. Returns true when the context is
     *        ready and the frame should proceed; false if still initializing or
     *        failed (caller should return early).
     */
    [[nodiscard]] bool ensure_webgpu_ready();

    /**
     * @brief Resize the WebGPU surface if the framebuffer was resized.
     *
     * Subclasses that override run_one_frame() should call this at the end
     * of each frame to keep the surface in sync with the window size.
     */
    void handle_framebuffer_resize();

   protected:
    glm::vec2 m_mouse_scroll_delta;
    glm::vec2 m_mouse_pos;
    std::optional<glm::vec2> m_last_mouse_pos{std::nullopt};
    std::bitset<ImGuiMouseButton_COUNT> m_mouse_states{};
    std::bitset<ImGuiKey_COUNT> m_key_states{};
    std::array<std::string_view, ImGuiMouseButton_COUNT> m_mouse_initiated_window{};
    std::array<std::string_view, ImGuiKey_COUNT> m_key_initiated_window{};

    std::unordered_map<std::string_view, ImGuiWindowInfo> m_imgui_window_info;

    std::string_view m_cur_hovered_widget, m_prev_hovered_widget;
    std::string_view m_cur_focused_widget;

    static constexpr auto k_no_hovered_widget = "";

    void run_one_frame() override;

   private:
    void init_windowing();

    double m_last_frame_time{0.0};
    bool m_first_loop_done{false};

    // Windowing and rendering members
    std::unique_ptr<pts::rendering::IWindowing> m_windowing;
    std::unique_ptr<pts::rendering::IViewport> m_viewport;
    std::unique_ptr<pts::rendering::WebGpuContext> m_webgpu_context;
    bool m_framebuffer_resized{false};

    // Class invariants (after init_windowing()):
    // - m_windowing is valid (non-null)
    // - m_viewport is valid (non-null)
    // - m_webgpu_context is non-null; may be Initializing,
    //   guaranteed Ready before loop() is called (driven by run_one_frame())
    // - m_imgui_windowing is valid (non-null)
    // - m_imgui_rendering is null until WebGPU context is ready;
    //   guaranteed valid before loop() is called (created in run_one_frame)
    void ensure_imgui_rendering();
    std::unique_ptr<pts::rendering::IImguiWindowing> m_imgui_windowing;
    std::unique_ptr<pts::rendering::IImguiRendering> m_imgui_rendering;
};
}  // namespace pts
