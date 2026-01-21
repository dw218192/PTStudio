#pragma once

#include <core/guiApplication.h>
#include <core/inputAction.h>
#include <core/legacy/archive.h>
#include <core/legacy/camera.h>
#include <core/legacy/scene.h>
#include <core/loggingManager.h>
#include <core/pluginManager.h>
#include <core/renderConfig.h>
#include <core/rendering/plugin.h>
#include <core/signal.h>
#include <spdlog/sinks/ringbuffer_sink.h>

#include <array>
#include <cstdlib>
#include <iostream>

#include "imgui/includes.h"

namespace pts::editor {
constexpr auto k_init_move_sensitivity = 5.0f;
constexpr auto k_init_rot_sensitivity = 60.0f;
constexpr auto k_object_select_mouse_time = 1.0f;

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
    auto on_scene_opened(PTS::Scene& scene) -> void;
    auto on_render_config_change(RenderConfig const& conf) -> void;
    auto on_mouse_leave_scene_viewport() noexcept -> void;
    auto on_mouse_enter_scene_viewport() noexcept -> void;

    // other helpers
    auto try_select_object() noexcept -> void;
    auto handle_input(InputEvent const& event) noexcept -> void override;
    auto on_remove_object(PTS::Ref<PTS::SceneObject> obj) -> void;
    auto on_add_oject(PTS::Ref<PTS::SceneObject> obj) -> void;

    AppConfig m_app_config;

    std::string m_console_text;
    std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> m_console_log_sink;

    // rendering
    RenderConfig m_config;
    PTS::Scene m_scene;
    PTS::Camera m_cam;

    // input handling
    std::vector<InputAction> m_input_actions;

    std::unique_ptr<PTS::Archive> m_archive;
    PluginHandle m_renderer_plugin{nullptr};
    RendererPluginInterfaceV1* m_renderer_interface{nullptr};
    PtsHostApi m_renderer_host_api{};
    uint64_t m_frame_index{0};

    struct ControlState {
        using ObjChangeCallback = std::function<void(PTS::SceneObject*)>;

        auto set_cur_obj(PTS::SceneObject* obj) noexcept -> void;
        auto get_cur_obj() const noexcept {
            return m_cur_obj;
        }

        auto get_on_selected_obj_change_callback_list() -> auto& {
            return m_on_selected_obj_change_callback_list;
        }

        float move_sensitivity = k_init_move_sensitivity;
        float rot_sensitivity = k_init_rot_sensitivity;
        std::array<char, 1024> obj_name_buf{};
        bool is_outside_view{false};
        struct GizmoState {
            ImGuizmo::OPERATION op{ImGuizmo::OPERATION::TRANSLATE};
            ImGuizmo::MODE mode{ImGuizmo::MODE::WORLD};
            bool snap{false};
            glm::vec3 snap_scale{1.0};
        } gizmo_state{};

        bool unlimited_fps{true};
        int is_changing_scene_cam{0};

       private:
        PTS::SceneObject* m_cur_obj{nullptr};
        pts::Signal<void(PTS::SceneObject*)> m_on_selected_obj_change_callback_list;
    } m_control_state;
};
}  // namespace pts::editor
