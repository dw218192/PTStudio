#include <core/guiApplication.h>
#include <core/imgui/imhelper.h>
#include <imgui_internal.h>

#include <stdexcept>

#include "guiApplicationImpl.h"

#if !defined(PTS_WINDOWING_GLFW)
#define PTS_WINDOWING_GLFW 1
#endif

namespace pts {
GUIApplication::GUIApplication(std::string_view name, pts::LoggingManager& logging_manager,
                               pts::PluginManager& plugin_manager, unsigned width, unsigned height,
                               float min_frame_time)
    : Application{name, logging_manager, plugin_manager} {
    set_min_frame_time(min_frame_time);
    m_impl = create_gui_application_impl(*this, name, logging_manager, width, height);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // Enable Docking

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    m_windowing = m_impl->create_windowing();
    m_rendering_host =
        std::make_unique<pts::rendering::Rendering>(*m_windowing, get_logging_manager());
}

GUIApplication::~GUIApplication() {
    m_rendering_host.reset();
    m_windowing.reset();
    ImGui::DestroyContext();
    m_impl.reset();
}

void GUIApplication::run() {
    static bool s_once = false;
    double last_frame_time = 0;
    while (!m_impl->should_close()) {
        auto const now = m_impl->time();

        // Poll and handle events (inputs, window resize, etc.)
        m_mouse_scroll_delta = glm::vec2{0.0f};

        m_impl->poll_events();
        poll_input_events();

        m_delta_time = static_cast<float>(now - last_frame_time);

        if (m_delta_time >= m_min_frame_time) {
            m_prev_hovered_widget = m_cur_hovered_widget;
            m_cur_hovered_widget = "";
            m_cur_focused_widget = "";

            // Start the Dear ImGui frame
            m_rendering_host->new_frame();
            ImGui::NewFrame();

            if (!s_once) {
                on_begin_first_loop();
                s_once = true;
            }

            // User Rendering
            loop(m_delta_time);

            // Process debug drawing events
            get_debug_drawer().loop(*this, m_delta_time);

            ImGui::Render();
            m_rendering_host->render(m_framebuffer_resized);
            m_framebuffer_resized = false;
            last_frame_time = now;

            // process hover change events
            if (m_prev_hovered_widget != m_cur_hovered_widget) {
                if (m_prev_hovered_widget != k_no_hovered_widget) {
                    // call on_leave_region on the previous widget
                    auto it = m_imgui_window_info.find(m_prev_hovered_widget);
                    if (it != m_imgui_window_info.end()) {
                        it->second.on_leave_region();
                    }
                }

                // call on_enter_region on the current widget
                auto it = m_imgui_window_info.find(m_cur_hovered_widget);
                if (it != m_imgui_window_info.end()) {
                    it->second.on_enter_region();
                }
            }
        }
    }
}

auto GUIApplication::get_render_graph_api() const noexcept -> const PtsRenderGraphApi* {
    return m_rendering_host ? m_rendering_host->render_graph_api() : nullptr;
}

auto GUIApplication::get_render_output_texture() const noexcept -> PtsTexture {
    return m_rendering_host ? m_rendering_host->output_texture() : PtsTexture{};
}

auto GUIApplication::get_render_output_imgui_id() const noexcept -> ImTextureID {
    return m_rendering_host ? m_rendering_host->output_imgui_id() : nullptr;
}

auto GUIApplication::resize_render_output(uint32_t width, uint32_t height) -> void {
    if (m_rendering_host) {
        m_rendering_host->resize_render_graph(width, height);
    }
}

auto GUIApplication::set_render_graph_current() -> void {
    if (m_rendering_host) {
        m_rendering_host->set_render_graph_current();
    }
}

auto GUIApplication::clear_render_graph_current() -> void {
    if (m_rendering_host) {
        m_rendering_host->clear_render_graph_current();
    }
}

auto GUIApplication::on_begin_first_loop() -> void {
}

auto GUIApplication::poll_input_events() noexcept -> void {
    auto screen_dim = glm::ivec2{get_window_width(), get_window_height()};
    double x = 0.0;
    double y = 0.0;
    m_impl->cursor_pos(x, y);
    if (!m_last_mouse_pos) {
        m_last_mouse_pos = m_mouse_pos = {x, y};
    } else {
        m_last_mouse_pos = m_mouse_pos;
        m_mouse_pos = {x, y};
    }

    // key events
    for (int i = 0; i < m_key_states.size(); ++i) {
        std::optional<Input> input;
        auto key_state = ImGui::IsKeyDown(static_cast<ImGuiKey>(i));
        if (key_state) {
            if (m_key_states[i]) {
                input = Input{InputType::KEYBOARD, ActionType::HOLD, i};
            } else {
                input = Input{InputType::KEYBOARD, ActionType::PRESS, i};
                m_key_initiated_window[i] = m_cur_hovered_widget;
            }
        } else {
            if (m_key_states[i]) {
                input = Input{InputType::KEYBOARD, ActionType::RELEASE, i};
            }
        }
        if (input) {
            auto event = InputEvent{*input,     m_mouse_pos,          *m_last_mouse_pos,
                                    screen_dim, m_mouse_scroll_delta, m_cur_hovered_widget,
                                    get_time()};
            handle_input(event);
            if (input->action_type == ActionType::RELEASE) {
                m_key_initiated_window[i] = k_no_hovered_widget;
            }
        }
        m_key_states[i] = key_state;
    }

    // mouse events

    // scroll
    if (glm::length(m_mouse_scroll_delta) > 0) {
        auto input = Input{InputType::MOUSE, ActionType::SCROLL, m_impl->middle_mouse_button()};
        handle_input(InputEvent{input, m_mouse_pos, screen_dim, m_mouse_scroll_delta,
                                m_mouse_initiated_window[ImGuiMouseButton_Middle], get_time()});
    }

    for (int i = 0; i < m_mouse_states.size(); ++i) {
        std::optional<Input> input;
        auto mouse_state = ImGui::IsMouseDown(i);
        if (mouse_state) {
            if (m_mouse_states[i]) {
                input = Input{InputType::MOUSE, ActionType::HOLD, i};
            } else {
                input = Input{InputType::MOUSE, ActionType::PRESS, i};
                m_mouse_initiated_window[i] = m_cur_hovered_widget;
            }
        } else {
            if (m_mouse_states[i]) {
                input = Input{InputType::MOUSE, ActionType::RELEASE, i};
            }
        }

        if (input) {
            auto event = InputEvent{*input,     m_mouse_pos,          *m_last_mouse_pos,
                                    screen_dim, m_mouse_scroll_delta, m_mouse_initiated_window[i],
                                    get_time()};
            handle_input(event);
            if (input->action_type == ActionType::RELEASE) {
                m_mouse_initiated_window[i] = k_no_hovered_widget;
            }
        }
        m_mouse_states[i] = mouse_state;
    }
}

void GUIApplication::on_scroll_event(double x, double y) noexcept {
    m_mouse_scroll_delta = {x, y};
}

void GUIApplication::on_framebuffer_resized() noexcept {
    m_framebuffer_resized = true;
}

auto GUIApplication::get_window_extent() const noexcept -> glm::ivec2 {
    return m_impl->window_extent();
}

auto GUIApplication::set_cursor_pos(float x, float y) noexcept -> void {
    m_impl->set_cursor_pos(x, y);
}

auto GUIApplication::get_window_height() const noexcept -> int {
    auto const extent = m_windowing->framebuffer_extent();
    return static_cast<int>(extent.height);
}

auto GUIApplication::get_window_width() const noexcept -> int {
    auto const extent = m_windowing->framebuffer_extent();
    return static_cast<int>(extent.width);
}

auto GUIApplication::begin_imgui_window(std::string_view name, ImGuiWindowFlags flags) noexcept
    -> bool {
    auto const ret = ImGui::Begin(name.data(), nullptr, flags);
    if (ImGui::IsWindowHovered(ImGuiItemStatusFlags_HoveredRect)) {
        m_cur_hovered_widget = name;
    }
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows)) {
        m_cur_focused_widget = name;
    }
    return ret;
}

void GUIApplication::end_imgui_window() noexcept {
    ImGui::End();
}

auto GUIApplication::get_window_content_pos(std::string_view name) const noexcept
    -> std::optional<ImVec2> {
    auto const win = ImGui::FindWindowByName(name.data());
    if (!win) {
        return std::nullopt;
    }
    return win->ContentRegionRect.Min;
}

float GUIApplication::get_time() const noexcept {
    return static_cast<float>(m_impl->time());
}

float GUIApplication::get_delta_time() const noexcept {
    return m_delta_time;
}
}  // namespace pts
