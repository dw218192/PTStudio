#pragma once

#include <core/application.h>
#include <core/inputAction.h>
#include <core/legacy/debugDrawer.h>
#include <core/rendering/rendering.h>
#include <core/rendering/windowing.h>
#include <core/signal.h>
#include <imgui.h>

#include <array>
#include <bitset>
#include <functional>
#include <glm/glm.hpp>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pts {
/**
 * @brief GUI application. Responsible for creating the window and polling events.
 */
struct GUIApplication : Application {
    struct Impl;

    // used to help detect if the mouse enters/leaves certain imgui windows
    struct ImGuiWindowInfo {
        Signal<void()> on_leave_region;
        Signal<void()> on_enter_region;
    };

    NO_COPY_MOVE(GUIApplication);

    GUIApplication(std::string_view name, pts::LoggingManager& logging_manager,
                   pts::PluginManager& plugin_manager, unsigned width, unsigned height,
                   float min_frame_time);
    ~GUIApplication() override;

    void run() override;

    [[nodiscard]] auto get_window_width() const noexcept -> int;
    [[nodiscard]] auto get_window_height() const noexcept -> int;

    void on_scroll_event(double x, double y) noexcept;
    void on_framebuffer_resized() noexcept;
    /**
     * @brief Called every frame. Override to handle the main loop.
     * @param dt the time since the last frame
     */
    virtual void loop(float dt) = 0;

    [[nodiscard]] auto get_render_graph_api() const noexcept -> const PtsRenderGraphApi*;
    [[nodiscard]] auto get_render_output_texture() const noexcept -> PtsTexture;
    [[nodiscard]] auto get_render_output_imgui_id() const noexcept -> ImTextureID;
    auto resize_render_output(uint32_t width, uint32_t height) -> void;
    auto set_render_graph_current() -> void;
    auto clear_render_graph_current() -> void;

   protected:
    virtual auto handle_input(InputEvent const& event) noexcept -> void {
    }
    virtual auto on_begin_first_loop() -> void;
    auto poll_input_events() noexcept -> void;
    [[nodiscard]] auto get_window_extent() const noexcept -> glm::ivec2;
    auto set_cursor_pos(float x, float y) noexcept -> void;

    /**
     * @brief Gets the renderer for the application.
     * @return the renderer
     */
    [[nodiscard]] auto get_debug_drawer() -> PTS::DebugDrawer& {
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

    PTS::DebugDrawer m_debug_drawer;
    float m_min_frame_time;
    float m_delta_time{0.0f};
    std::unordered_map<std::string_view, ImGuiWindowInfo> m_imgui_window_info;

    std::string_view m_cur_hovered_widget, m_prev_hovered_widget;
    std::string_view m_cur_focused_widget;

    static constexpr auto k_no_hovered_widget = "";

    std::unique_ptr<pts::rendering::IWindowing> m_windowing;
    std::unique_ptr<pts::rendering::Rendering> m_rendering_host;
    bool m_framebuffer_resized{false};

   private:
    std::unique_ptr<Impl> m_impl;
};
}  // namespace pts
