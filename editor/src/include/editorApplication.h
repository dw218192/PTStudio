#pragma once

#include <core/legacy/archive.h>
#include <core/legacy/callbackList.h>
#include <core/legacy/camera.h>
#include <core/legacy/renderConfig.h>
#include <core/legacy/scene.h>
#include <core/loggingManager.h>
#include <core/pluginManager.h>
#include <core/rendering/plugin.h>
#include <spdlog/sinks/ringbuffer_sink.h>

#include <array>
#include <cstdlib>
#include <iostream>

#include <core/glfwApplication.h>
#include <core/inputAction.h>

#include "imgui/includes.h"

namespace PTS::Editor {
constexpr auto k_init_move_sensitivity = 5.0f;
constexpr auto k_init_rot_sensitivity = 60.0f;
constexpr auto k_object_select_mouse_time = 1.0f;

struct EditorApplication final : GLFWApplication {
    NO_COPY_MOVE(EditorApplication);

    auto loop(float dt) -> void override;
    auto quit(int code) -> void override;
    /**
     * @brief Checks a result returned from some function. Prints the error and Terminates the
     * program on error.
     * @tparam T Type of the real return value
     * @tparam E Type of the error return value
     * @param res the result
     * @return The real return value if no error
     */
    template <typename T, typename E>
    static constexpr auto check_error(tl::expected<T, E> const& res) -> decltype(auto);
    template <typename T, typename E>
    static constexpr auto check_error(tl::expected<T, E>&& res) -> decltype(auto);

   protected:
    auto on_begin_first_loop() -> void override;

   public:
    EditorApplication(std::string_view name, RenderConfig config,
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
    auto on_scene_opened(Scene& scene) -> void;
    auto on_render_config_change(RenderConfig const& conf) -> void;
    auto on_mouse_leave_scene_viewport() noexcept -> void;
    auto on_mouse_enter_scene_viewport() noexcept -> void;

    // other helpers
    auto try_select_object() noexcept -> void;
    auto handle_input(InputEvent const& event) noexcept -> void override;
    auto on_remove_object(Ref<SceneObject> obj) -> void;
    auto on_add_oject(Ref<SceneObject> obj) -> void;

    std::string m_console_text;
    std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> m_console_log_sink;

    // rendering
    RenderConfig m_config;
    Scene m_scene;
    Camera m_cam;

    // input handling
    std::vector<InputAction> m_input_actions;

    std::unique_ptr<Archive> m_archive;
    PluginHandle m_renderer_plugin{nullptr};
    RendererPluginInterfaceV1* m_renderer_interface{nullptr};
    PtsHostApi m_renderer_host_api{};
    uint64_t m_frame_index{0};

    struct ControlState {
        using ObjChangeCallback = std::function<void(ObserverPtr<SceneObject>)>;

        auto set_cur_obj(ObserverPtr<SceneObject> obj) noexcept -> void;
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
        ObserverPtr<SceneObject> m_cur_obj{nullptr};
        CallbackList<void(ObserverPtr<SceneObject>)> m_on_selected_obj_change_callback_list;
    } m_control_state;
};

template <typename T, typename E>
constexpr auto EditorApplication::check_error(tl::expected<T, E> const& res) -> decltype(auto) {
    if (!res) {
        std::cerr << res.error() << std::endl;
        std::exit(-1);
    }
    if constexpr (!std::is_void_v<T>) {
        return res.value();
    }
}

template <typename T, typename E>
constexpr auto EditorApplication::check_error(tl::expected<T, E>&& res) -> decltype(auto) {
    if (!res) {
        std::cerr << res.error() << std::endl;
        std::exit(-1);
    }
    if constexpr (!std::is_void_v<T>) {
        return std::move(res).value();
    }
}
}  // namespace PTS::Editor
