#pragma once

#include <core/defines.h>
#include <core/inputAction.h>
#include <imgui.h>

#include <array>
#include <bitset>
#include <functional>
#include <glm/glm.hpp>
#include <optional>
#include <string_view>

namespace pts {
namespace rendering {
class IViewport;
}

class InputComponent {
   public:
    NO_COPY_MOVE(InputComponent);

    explicit InputComponent(rendering::IViewport& viewport);
    ~InputComponent();

    using InputHandler = std::function<void(const InputEvent&)>;
    void set_handler(InputHandler handler);

    /// Poll input state from ImGui. Call once per frame after ImGui::NewFrame().
    void poll(float time, int window_width, int window_height,
              std::string_view cur_hovered_widget);

    [[nodiscard]] auto mouse_pos() const noexcept -> glm::vec2;
    [[nodiscard]] auto mouse_scroll_delta() const noexcept -> glm::vec2;

    void on_scroll_event(double x, double y) noexcept;

    void reset_scroll_delta() noexcept;

   private:
    rendering::IViewport& m_viewport;
    InputHandler m_handler;

    glm::vec2 m_mouse_scroll_delta{0.0f};
    glm::vec2 m_mouse_pos{0.0f};
    std::optional<glm::vec2> m_last_mouse_pos{std::nullopt};
    std::bitset<ImGuiMouseButton_COUNT> m_mouse_states{};
    std::bitset<ImGuiKey_COUNT> m_key_states{};
    std::array<std::string_view, ImGuiMouseButton_COUNT> m_mouse_initiated_window{};
    std::array<std::string_view, ImGuiKey_COUNT> m_key_initiated_window{};

    static constexpr auto k_no_hovered_widget = "";
};

}  // namespace pts
