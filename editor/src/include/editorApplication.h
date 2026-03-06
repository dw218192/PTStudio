#pragma once

#include <core/windowedApplication.h>
#include <core/inputAction.h>
#include <spdlog/sinks/ringbuffer_sink.h>

#include <memory>
#include <vector>

namespace pts {
class ImGuiComponent;
class InputComponent;
}  // namespace pts

namespace pts::editor {
struct AppConfig {
    bool quit_on_start{false};
};

struct EditorApplication final : WindowedApplication {
    NO_COPY_MOVE(EditorApplication);

    EditorApplication(std::string_view name, pts::LoggingManager& logging_manager);
    ~EditorApplication() override;

    void register_args(CommandLine& cli) override;
    void process_args(const CommandLine& cli) override;

   protected:
    void on_ready() override;
    void update(float dt) override;
    void render(FrameContext& ctx) override;

   private:
    void setup_docking_layout();
    auto create_input_actions() noexcept -> void;
    auto wrap_mouse_pos() noexcept -> void;

    // imgui rendering
    auto draw_scene_panel() noexcept -> void;
    auto draw_object_panel() noexcept -> void;
    auto draw_scene_viewport() noexcept -> void;
    auto draw_console_panel() const noexcept -> void;

    // events
    auto on_mouse_leave_scene_viewport() noexcept -> void;
    auto on_mouse_enter_scene_viewport() noexcept -> void;

    auto handle_input(InputEvent const& event) noexcept -> void;

    // Components
    std::unique_ptr<ImGuiComponent> m_imgui;
    std::unique_ptr<InputComponent> m_input;

    AppConfig m_app_config;

    std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> m_console_log_sink;

    float m_fovy = 60.0f;

    // input handling
    std::vector<InputAction> m_input_actions;

    bool m_first_frame{true};
};
}  // namespace pts::editor
