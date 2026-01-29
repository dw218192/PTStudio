#pragma once

#include <core/guiApplication.h>
#include <core/inputAction.h>
#include <core/loggingManager.h>
#include <core/pluginManager.h>
#include <core/renderConfig.h>
#include <core/rendering/plugin.h>
#include <core/rendering/webgpu/webgpu.h>
#include <core/signal.h>
#include <spdlog/sinks/ringbuffer_sink.h>

#include <array>
#include <cstdlib>
#include <iostream>

#include "imgui/includes.h"

namespace pts::editor {
struct AppConfig {
    bool quit_on_start{false};
};

struct EditorApplication final : GUIApplication {
    NO_COPY_MOVE(EditorApplication);

    auto loop(float dt) -> void override;

   protected:
    auto on_begin_first_loop() -> void override;

   public:
    EditorApplication(std::string_view name, RenderConfig config, AppConfig app_config,
                      pts::LoggingManager& logging_manager, pts::PluginManager& plugin_manager);
    ~EditorApplication() override;

    auto create_input_actions() noexcept -> void;
    auto wrap_mouse_pos() noexcept -> void;

    // imgui rendering
    auto draw_scene_panel() noexcept -> void;
    auto draw_object_panel() noexcept -> void;
    auto draw_scene_viewport() noexcept -> void;
    auto draw_console_panel() const noexcept -> void;

    // events
    auto on_render_config_change(RenderConfig const& conf) -> void;
    auto on_mouse_leave_scene_viewport() noexcept -> void;
    auto on_mouse_enter_scene_viewport() noexcept -> void;

    auto handle_input(InputEvent const& event) noexcept -> void override;

    AppConfig m_app_config;

    std::string m_console_text;
    std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> m_console_log_sink;

    // rendering
    RenderConfig m_config;

    // input handling
    std::vector<InputAction> m_input_actions;

    PluginHandle m_renderer_plugin{nullptr};
    RendererPluginInterfaceV1* m_renderer_interface{nullptr};
    PtsHostApi m_renderer_host_api{};
    uint64_t m_frame_index{0};
};
}  // namespace pts::editor
