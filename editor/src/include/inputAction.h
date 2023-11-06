#pragma once
#include <functional>
#include "utils.h"

namespace PTS {
namespace Editor {

enum class InputType {
    KEYBOARD,
    MOUSE
};
enum class ActionType {
    PRESS,
    RELEASE
};

struct InputAction {
    DEFAULT_COPY_MOVE(InputAction);
    ~InputAction() = default;
    InputAction(InputType input_type, ActionType action_type, int key_or_button, std::function<void()> action) noexcept;
    auto add_constraint(std::function<bool()> constraint) noexcept -> InputAction&;
    auto operator()() const noexcept -> void;
private:
    InputType m_input_type;
    ActionType m_action_type;
    int m_key_or_button;
    std::function<void()> m_action;
    std::vector<std::function<bool()>> m_constraints;
};

}
}