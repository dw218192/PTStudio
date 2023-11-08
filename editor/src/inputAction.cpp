#include "include/inputAction.h"

using namespace PTS;

InputAction::InputAction(Input input,
                         std::function<void(InputEvent const&)> action) noexcept : 
m_input{std::move(input)}, m_action{std::move(action)} {}

auto InputAction::operator()(InputEvent const& event) const noexcept -> void {
	if (m_input != event.input) {
		return;
	}
	
	for (const auto& constraint : m_constraints) {
		if (!constraint(event)) {
			return;
		}
	}
	m_action(event);
}
