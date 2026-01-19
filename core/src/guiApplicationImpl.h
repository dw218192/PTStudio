#pragma once

#include <core/guiApplication.h>
#include <core/rendering/windowing.h>

#include <memory>

namespace pts {
struct GUIApplication::Impl {
    virtual ~Impl() = default;

    [[nodiscard]] virtual auto create_windowing() const
        -> std::unique_ptr<rendering::IWindowing> = 0;
    virtual void poll_events() const = 0;
    [[nodiscard]] virtual auto should_close() const -> bool = 0;
    [[nodiscard]] virtual auto time() const noexcept -> double = 0;
    [[nodiscard]] virtual auto window_extent() const noexcept -> glm::ivec2 = 0;
    virtual void set_cursor_pos(float x, float y) noexcept = 0;
    virtual void cursor_pos(double& x, double& y) const = 0;
    [[nodiscard]] virtual auto middle_mouse_button() const noexcept -> int = 0;
};

// implemented by the windowing implementations
[[nodiscard]] auto create_gui_application_impl(GUIApplication& app, std::string_view name,
                                               LoggingManager& logging_manager, unsigned width,
                                               unsigned height)
    -> std::unique_ptr<GUIApplication::Impl>;

}  // namespace pts
