#pragma once

#include <core/application.h>
#include <core/inputAction.h>
#include <core/rendering/graph.h>
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
namespace rendering {
class IImguiRendering;
class IImguiWindowing;
class IRenderGraph;
}  // namespace rendering

/**
 * @brief GUI application with ImGui support. Extends Application with UI capabilities.
 */
struct GUIApplication : Application {
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

    void on_scroll_event(double x, double y) noexcept;
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

    float m_min_frame_time;
    std::unordered_map<std::string_view, ImGuiWindowInfo> m_imgui_window_info;

    std::string_view m_cur_hovered_widget, m_prev_hovered_widget;
    std::string_view m_cur_focused_widget;

    static constexpr auto k_no_hovered_widget = "";

    // Class invariants (enforced in constructor, throw on failure):
    // - m_render_graph is always valid (non-null)
    // - m_imgui_windowing is always valid (non-null)
    // - m_imgui_rendering is always valid (non-null)
    std::unique_ptr<pts::rendering::IRenderGraph> m_render_graph;
    std::unique_ptr<pts::rendering::IImguiWindowing> m_imgui_windowing;
    std::unique_ptr<pts::rendering::IImguiRendering> m_imgui_rendering;
};
}  // namespace pts
