#include <core/components/inputComponent.h>
#include <core/rendering/windowing.h>

namespace pts {

InputComponent::InputComponent(rendering::IViewport& viewport) : m_viewport{viewport} {
    m_scroll_connection =
        m_viewport.on_scroll.connect([this](double dx, double dy) { on_scroll_event(dx, dy); });
}

InputComponent::~InputComponent() = default;

void InputComponent::set_handler(InputHandler handler) {
    m_handler = std::move(handler);
}

void InputComponent::poll(float time, int window_width, int window_height,
                          std::string_view cur_hovered_widget) {
    // Snapshot and reset scroll delta atomically — poll() consumes accumulated scroll
    auto scroll_delta = m_mouse_scroll_delta;
    m_mouse_scroll_delta = glm::vec2{0.0f};

    auto screen_dim = glm::ivec2{window_width, window_height};
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
    for (ImGuiKey key = ImGuiKey_NamedKey_BEGIN; key < ImGuiKey_NamedKey_END;
         key = static_cast<ImGuiKey>(key + 1)) {
        auto const key_index = static_cast<size_t>(key - ImGuiKey_NamedKey_BEGIN);
        std::optional<Input> input;
        auto const key_state = ImGui::IsKeyDown(key);
        if (key_state) {
            if (m_key_states[key_index]) {
                input = Input{InputType::KEYBOARD, ActionType::HOLD, static_cast<int>(key)};
            } else {
                input = Input{InputType::KEYBOARD, ActionType::PRESS, static_cast<int>(key)};
                m_key_initiated_window[key_index] = cur_hovered_widget;
            }
        } else if (m_key_states[key_index]) {
            input = Input{InputType::KEYBOARD, ActionType::RELEASE, static_cast<int>(key)};
        }
        if (input && m_handler) {
            auto event = InputEvent{*input,     m_mouse_pos,  *m_last_mouse_pos,
                                    screen_dim, scroll_delta, cur_hovered_widget,
                                    time};
            m_handler(event);
            if (input->action_type == ActionType::RELEASE) {
                m_key_initiated_window[key_index] = k_no_hovered_widget;
            }
        }
        m_key_states[key_index] = key_state;
    }

    // mouse events

    // scroll
    if (glm::length(scroll_delta) > 0 && m_handler) {
        auto input = Input{InputType::MOUSE, ActionType::SCROLL, ImGuiMouseButton_Middle};
        m_handler(
            InputEvent{input, m_mouse_pos, screen_dim, scroll_delta, cur_hovered_widget, time});
    }

    for (int i = 0; i < static_cast<int>(m_mouse_states.size()); ++i) {
        std::optional<Input> input;
        auto mouse_state = ImGui::IsMouseDown(i);
        if (mouse_state) {
            if (m_mouse_states[i]) {
                input = Input{InputType::MOUSE, ActionType::HOLD, i};
            } else {
                input = Input{InputType::MOUSE, ActionType::PRESS, i};
                m_mouse_initiated_window[i] = cur_hovered_widget;
            }
        } else {
            if (m_mouse_states[i]) {
                input = Input{InputType::MOUSE, ActionType::RELEASE, i};
            }
        }

        if (input && m_handler) {
            auto event = InputEvent{*input,     m_mouse_pos,  *m_last_mouse_pos,
                                    screen_dim, scroll_delta, m_mouse_initiated_window[i],
                                    time};
            m_handler(event);
            if (input->action_type == ActionType::RELEASE) {
                m_mouse_initiated_window[i] = k_no_hovered_widget;
            }
        }
        m_mouse_states[i] = mouse_state;
    }
}

void InputComponent::on_scroll_event(double x, double y) noexcept {
    m_mouse_scroll_delta += glm::vec2{x, y};
}

auto InputComponent::mouse_pos() const noexcept -> glm::vec2 {
    return m_mouse_pos;
}

auto InputComponent::mouse_scroll_delta() const noexcept -> glm::vec2 {
    return m_mouse_scroll_delta;
}

}  // namespace pts
