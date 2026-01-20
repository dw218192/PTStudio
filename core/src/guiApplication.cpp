#include <core/guiApplication.h>
#include <core/imgui/imhelper.h>
#include <imgui_internal.h>

#include <chrono>

#include "rendering/imguiBackend.h"
#include "rendering/renderingComponents.h"

namespace pts {
namespace {
auto time_since_start(const std::chrono::steady_clock::time_point& start) -> double {
    auto const now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start).count();
}
}  // namespace

GUIApplication::GUIApplication(std::string_view name, pts::LoggingManager& logging_manager,
                               pts::PluginManager& plugin_manager, unsigned width, unsigned height,
                               float min_frame_time)
    : Application{name, logging_manager, plugin_manager} {
    set_min_frame_time(min_frame_time);
    m_start_time = std::chrono::steady_clock::now();

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // Enable Docking

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    m_windowing = pts::rendering::create_windowing(get_logging_manager());
    auto viewport_desc = pts::rendering::ViewportDesc{
        get_name().data(), width, height, true, true, true, true,
    };
    m_viewport = m_windowing->create_viewport(viewport_desc);
    m_viewport->on_scroll.connect([this](double dx, double dy) { on_scroll_event(dx, dy); });
    m_viewport->on_drawable_resized.connect(
        [this](pts::rendering::Extent2D) { on_framebuffer_resized(); });
    m_viewport->on_close_requested.connect([this]() {
        if (m_viewport) {
            m_viewport->request_close();
        }
    });

    m_imgui_windowing = pts::rendering::create_imgui_windowing(*m_viewport, get_logging_manager());

    auto rendering_components = pts::rendering::create_rendering_components(
        *m_windowing, *m_viewport, get_logging_manager());
    m_render_graph = std::move(rendering_components.render_graph);
    m_imgui_rendering = std::move(rendering_components.imgui_rendering);

    auto const extent = m_viewport->drawable_extent();
    resize_render_output(extent.w, extent.h);
}

GUIApplication::~GUIApplication() {
    m_imgui_rendering.reset();
    m_imgui_windowing.reset();
    m_render_graph.reset();
    m_viewport.reset();
    m_windowing.reset();
    ImGui::DestroyContext();
}

void GUIApplication::run() {
    static bool s_once = false;
    double last_frame_time = time_since_start(m_start_time);
    while (m_viewport && !m_viewport->should_close()) {
        auto const now = time_since_start(m_start_time);

        // Poll and handle events (inputs, window resize, etc.)
        m_mouse_scroll_delta = glm::vec2{0.0f};

        m_windowing->pump_events(pts::rendering::PumpEventMode::Poll);
        poll_input_events();

        m_delta_time = static_cast<float>(now - last_frame_time);

        if (m_delta_time >= m_min_frame_time) {
            m_prev_hovered_widget = m_cur_hovered_widget;
            m_cur_hovered_widget = "";
            m_cur_focused_widget = "";

            // Start the Dear ImGui frame
            if (m_imgui_windowing) {
                m_imgui_windowing->new_frame();
            }
            if (m_imgui_rendering) {
                m_imgui_rendering->new_frame();
            }
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
            if (m_imgui_rendering) {
                m_imgui_rendering->render(m_framebuffer_resized);
            }
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
    return m_render_graph ? m_render_graph->api() : nullptr;
}

auto GUIApplication::get_render_output_texture() const noexcept -> PtsTexture {
    return m_render_graph ? m_render_graph->output_texture() : PtsTexture{};
}

auto GUIApplication::get_render_output_imgui_id() const noexcept -> ImTextureID {
    return m_imgui_rendering ? m_imgui_rendering->output_id() : ImTextureID_Invalid;
}

auto GUIApplication::resize_render_output(uint32_t width, uint32_t height) -> void {
    if (!m_render_graph || !m_imgui_rendering) {
        return;
    }
    if (width == 0 || height == 0) {
        m_imgui_rendering->clear_render_output();
        return;
    }
    m_imgui_rendering->clear_render_output();
    m_render_graph->resize(width, height);
    m_imgui_rendering->set_render_output(*m_render_graph);
}

auto GUIApplication::set_render_graph_current() -> void {
    if (m_render_graph) {
        m_render_graph->set_current();
    }
}

auto GUIApplication::clear_render_graph_current() -> void {
    if (m_render_graph) {
        m_render_graph->clear_current();
    }
}

auto GUIApplication::on_begin_first_loop() -> void {
}

auto GUIApplication::poll_input_events() noexcept -> void {
    auto screen_dim = glm::ivec2{get_window_width(), get_window_height()};
    auto mouse_pos = ImGui::GetMousePos();
    if (!ImGui::IsMousePosValid(&mouse_pos)) {
        mouse_pos = ImVec2{0.0f, 0.0f};
    }
    double x = mouse_pos.x;
    double y = mouse_pos.y;
    if (!m_last_mouse_pos) {
        m_last_mouse_pos = m_mouse_pos = {x, y};
    } else {
        m_last_mouse_pos = m_mouse_pos;
        m_mouse_pos = {x, y};
    }

    // key events (keyboard only, ImGuiKey values)
    for (ImGuiKey key = ImGuiKey_NamedKey_BEGIN; key <= ImGuiKey_Oem102;
         key = static_cast<ImGuiKey>(key + 1)) {
        auto const key_index = static_cast<size_t>(key);
        std::optional<Input> input;
        auto const key_state = ImGui::IsKeyDown(key);
        if (key_state) {
            if (m_key_states[key_index]) {
                input = Input{InputType::KEYBOARD, ActionType::HOLD, static_cast<int>(key)};
            } else {
                input = Input{InputType::KEYBOARD, ActionType::PRESS, static_cast<int>(key)};
                m_key_initiated_window[key_index] = m_cur_hovered_widget;
            }
        } else if (m_key_states[key_index]) {
            input = Input{InputType::KEYBOARD, ActionType::RELEASE, static_cast<int>(key)};
        }
        if (input) {
            auto event = InputEvent{*input,     m_mouse_pos,          *m_last_mouse_pos,
                                    screen_dim, m_mouse_scroll_delta, m_cur_hovered_widget,
                                    get_time()};
            handle_input(event);
            if (input->action_type == ActionType::RELEASE) {
                m_key_initiated_window[key_index] = k_no_hovered_widget;
            }
        }
        m_key_states[key_index] = key_state;
    }

    // mouse events

    // scroll
    if (glm::length(m_mouse_scroll_delta) > 0) {
        auto input = Input{InputType::MOUSE, ActionType::SCROLL, ImGuiMouseButton_Middle};
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
    if (m_viewport) {
        auto const extent = m_viewport->drawable_extent();
        resize_render_output(extent.w, extent.h);
    }
}

auto GUIApplication::get_window_extent() const noexcept -> glm::ivec2 {
    if (!m_viewport) {
        return glm::ivec2{0, 0};
    }
    auto const extent = m_viewport->logical_extent();
    return glm::ivec2{static_cast<int>(extent.w), static_cast<int>(extent.h)};
}

auto GUIApplication::set_cursor_pos(float x, float y) noexcept -> void {
    if (m_viewport) {
        m_viewport->set_cursor_pos(x, y);
    }
}

auto GUIApplication::get_window_height() const noexcept -> int {
    if (!m_viewport) {
        return 0;
    }
    auto const extent = m_viewport->drawable_extent();
    return static_cast<int>(extent.h);
}

auto GUIApplication::get_window_width() const noexcept -> int {
    if (!m_viewport) {
        return 0;
    }
    auto const extent = m_viewport->drawable_extent();
    return static_cast<int>(extent.w);
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
    return static_cast<float>(time_since_start(m_start_time));
}

float GUIApplication::get_delta_time() const noexcept {
    return m_delta_time;
}
}  // namespace pts
