#pragma once
#include <core/utils.h>

#include <functional>
#include <glm/glm.hpp>
#include <optional>
#include <string_view>

namespace PTS {
enum class InputType { KEYBOARD, MOUSE };

enum class ActionType { PRESS, HOLD, RELEASE, SCROLL };

struct Input {
    InputType input_type;
    ActionType action_type;
    int key_or_button;

    auto operator==(Input const& other) const noexcept -> bool {
        return input_type == other.input_type && action_type == other.action_type &&
               key_or_button == other.key_or_button;
    }
    auto operator!=(Input const& other) const noexcept -> bool {
        return !(*this == other);
    }
};

struct InputEvent {
    InputEvent(Input input, glm::vec2 mouse_pos, glm::ivec2 screen_size,
               glm::vec2 mouse_scroll_delta, std::string_view initiated_window, float time) noexcept
        : input{std::move(input)},
          last_mouse_pos{mouse_pos},
          mouse_pos{mouse_pos},
          normalized_mouse_pos{mouse_pos / glm::vec2(screen_size)},
          mouse_scroll_delta{mouse_scroll_delta},
          initiated_window{initiated_window},
          time{time} {
    }

    InputEvent(Input input, glm::vec2 mouse_pos, glm::vec2 last_mouse_pos, glm::ivec2 screen_size,
               glm::vec2 mouse_scroll_delta, std::string_view initiated_window, float time) noexcept
        : input{std::move(input)},
          last_mouse_pos{last_mouse_pos},
          mouse_pos{mouse_pos},
          normalized_mouse_pos{mouse_pos / glm::vec2(screen_size)},
          mouse_delta{mouse_pos - last_mouse_pos},
          normalized_mouse_delta{mouse_delta / glm::vec2(screen_size)},
          mouse_scroll_delta{mouse_scroll_delta},
          initiated_window{initiated_window},
          time{time} {
    }

    Input input;
    // runtime info
    glm::vec2 last_mouse_pos;
    glm::vec2 mouse_pos;
    glm::vec2 normalized_mouse_pos;
    glm::vec2 mouse_delta{0.0f};
    glm::vec2 normalized_mouse_delta{0.0f};
    glm::vec2 mouse_scroll_delta{0.0f};
    std::string_view initiated_window;
    float time;
};

using InputActionConstraint = std::function<bool(InputEvent const&)>;

struct InputAction {
    DEFAULT_COPY_MOVE(InputAction);
    ~InputAction() = default;
    InputAction(Input input, std::function<void(InputEvent const&)> action) noexcept;
    auto operator()(InputEvent const& event) const noexcept -> void;

    auto get_input() const {
        return m_input;
    }

    auto add_constraint(InputActionConstraint constraint) & noexcept -> InputAction& {
        m_constraints.push_back(std::move(constraint));
        return *this;
    }

    auto add_constraint(InputActionConstraint constraint) && noexcept -> InputAction&& {
        m_constraints.push_back(std::move(constraint));
        return std::move(*this);
    }

   private:
    Input m_input;
    std::function<void(InputEvent const&)> m_action;
    std::vector<InputActionConstraint> m_constraints;
};
}  // namespace PTS
