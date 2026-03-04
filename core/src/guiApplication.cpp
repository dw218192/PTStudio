#include <core/diagnostics.h>
#include <core/guiApplication.h>
#include <core/rendering/webgpuContext.h>
#include <imgui_internal.h>

#include "rendering/imguiBackend.h"
#include "rendering/renderingComponents.h"

namespace pts {

GUIApplication::GUIApplication(std::string_view name, pts::LoggingManager& logging_manager,
                               unsigned width, unsigned height, float min_frame_time)
    : Application{name, logging_manager, min_frame_time}, m_min_frame_time{min_frame_time} {
    // Create windowing system
    m_windowing = pts::rendering::create_windowing(get_logging_manager());
    INVARIANT_MSG(m_windowing != nullptr, "create_windowing must return valid windowing system");

    // Create viewport
    auto viewport_desc = pts::rendering::ViewportDesc{
        get_name().data(), width, height, true, true, true, true,
    };
    m_viewport = m_windowing->create_viewport(viewport_desc);
    INVARIANT_MSG(m_viewport != nullptr, "create_viewport must return valid viewport");
    m_viewport->on_drawable_resized.connect(
        [this](pts::rendering::Extent2D) { on_framebuffer_resized(); });

    // Create WebGPU context (starts async initialization)
    m_webgpu_context = pts::rendering::WebGpuContext::create(*m_viewport, get_logging_manager());
    INVARIANT_MSG(m_webgpu_context != nullptr, "WebGpuContext::create must return valid context");

    // Setup Dear ImGui context (no WebGPU dependency)
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // Enable Docking

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup ImGui windowing backend
    m_viewport->on_scroll.connect([this](double dx, double dy) { on_scroll_event(dx, dy); });
    m_imgui_windowing = pts::rendering::create_imgui_windowing(*m_viewport, get_logging_manager());

    // ImGui rendering components are created lazily in ensure_imgui_rendering()
    // because the WebGPU context may still be initializing asynchronously.
}

GUIApplication::~GUIApplication() {
    m_imgui_rendering.reset();
    m_imgui_windowing.reset();
    ImGui::DestroyContext();
}

void GUIApplication::run() {
#if defined(__EMSCRIPTEN__)
    Application::run();
#else
    while (!m_viewport->should_close() && !should_stop()) {
        run_one_frame();
        check_frame_limit();
    }
#endif
}

bool GUIApplication::ensure_webgpu_ready() {
    // Already failed
    if (m_webgpu_context->is_failed()) {
        return false;
    }

    if (m_webgpu_context->is_initializing()) {
        m_webgpu_context->tick_init();

        if (m_webgpu_context->is_failed()) {
            log(pts::LogLevel::Error, "WebGPU context initialization failed");
            m_viewport->request_close();
            return false;
        }

        if (m_webgpu_context->is_initializing()) {
            return false;  // still initializing — wait for next frame
        }

        log(pts::LogLevel::Info, "Application initialized");
    }

    return true;
}

void GUIApplication::on_framebuffer_resized() noexcept {
    m_framebuffer_resized = true;
}

void GUIApplication::handle_framebuffer_resize() {
    if (m_framebuffer_resized) {
        auto const extent = m_viewport->drawable_extent();
        m_webgpu_context->surface().resize(extent);
        m_framebuffer_resized = false;
    }
}

auto GUIApplication::get_window_width() const noexcept -> int {
    return static_cast<int>(m_viewport->drawable_extent().w);
}

auto GUIApplication::get_window_height() const noexcept -> int {
    return static_cast<int>(m_viewport->drawable_extent().h);
}

void GUIApplication::ensure_imgui_rendering() {
    INVARIANT_MSG(m_webgpu_context != nullptr, "WebGPU context must be valid");

    auto imgui_components = pts::rendering::create_imgui_components(*m_webgpu_context, *m_viewport,
                                                                    get_logging_manager());
    INVARIANT_MSG(imgui_components.imgui_rendering != nullptr,
                  "create_imgui_components must return valid imgui_rendering");

    m_imgui_rendering = std::move(imgui_components.imgui_rendering);
}

void GUIApplication::run_one_frame() {
    auto const now = get_time();

    // Poll and handle events (inputs, window resize, etc.)
    m_mouse_scroll_delta = glm::vec2{0.0f};

    m_windowing->pump_events(pts::rendering::PumpEventMode::Poll);

    if (!ensure_webgpu_ready()) {
        return;
    }

    // Create ImGui rendering components once WebGPU context is ready
    if (!m_imgui_rendering) {
        ensure_imgui_rendering();
    }

    poll_input_events();

    float delta_time = static_cast<float>(now - m_last_frame_time);

    if (delta_time >= m_min_frame_time) {
        m_prev_hovered_widget = m_cur_hovered_widget;
        m_cur_hovered_widget = "";
        m_cur_focused_widget = "";

        // Start the Dear ImGui frame
        m_imgui_windowing->new_frame();
        m_imgui_rendering->new_frame();
        ImGui::NewFrame();

        if (!m_first_loop_done) {
            on_begin_first_loop();
            m_first_loop_done = true;
        }
        if (m_viewport && m_viewport->should_close()) {
            return;
        }

        // User Rendering
        loop(delta_time);

        ImGui::Render();
        m_imgui_rendering->render(false);  // Framebuffer resize handled in GraphicsApplication
        m_last_frame_time = now;

        // process hover change events
        if (m_prev_hovered_widget != m_cur_hovered_widget) {
            if (m_prev_hovered_widget != k_no_hovered_widget) {
                auto it = m_imgui_window_info.find(m_prev_hovered_widget);
                if (it != m_imgui_window_info.end()) {
                    it->second.on_leave_region();
                }
            }

            auto it = m_imgui_window_info.find(m_cur_hovered_widget);
            if (it != m_imgui_window_info.end()) {
                it->second.on_enter_region();
            }
        }
    }

    handle_framebuffer_resize();
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

auto GUIApplication::begin_imgui_window(std::string_view name, ImGuiWindowFlags flags) noexcept
    -> bool {
    auto const ret = ImGui::Begin(name.data(), nullptr, flags);
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
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

}  // namespace pts
