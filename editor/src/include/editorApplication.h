#pragma once

#include <imgui.h>
#include <ImGuizmo.h>
#include <array>
#include <iostream>

#include "editorRenderer.h"
#include "scene.h"

#include "glfwApplication.h"
#include "glTexture.h"
#include "singleton.h"
#include "archive.h"
#include "camera.h"
#include "inputAction.h"

namespace PTS {
namespace Editor {

constexpr auto k_init_move_sensitivity = 5.0f;
constexpr auto k_init_rot_sensitivity = 60.0f;
constexpr auto k_object_select_mouse_time = 1.0f;
constexpr auto k_default_renderer_idx = 0;

struct EditorApplication final : GLFWApplication, Singleton<EditorApplication> {
friend Singleton;
NO_COPY_MOVE(EditorApplication);

    void add_renderer(std::unique_ptr<Renderer> renderer) noexcept;
    void loop(float dt) override;
    void quit(int code) override;
    /**
	 * \brief Checks a result returned from some function. Prints the error and Terminates the program on error.
	 * \tparam T Type of the real return value
	 * \tparam E Type of the error return value
	 * \param res the result
	 * \return The real return value if no error
	 */
    template<typename T, typename E>
    static constexpr decltype(auto) check_error(tl::expected<T, E> const& res);
    template<typename T, typename E>
    static constexpr decltype(auto) check_error(tl::expected<T, E>&& res);

protected:
    void on_begin_first_loop() override;

private:
    EditorApplication(std::string_view name, RenderConfig config);
    ~EditorApplication() override = default;

    void create_input_actions() noexcept;

    // imgui rendering
    void draw_scene_panel() noexcept;
    void draw_object_panel() noexcept;
    void draw_scene_viewport(TextureHandle render_buf) noexcept;
    void draw_console_panel() const noexcept;

    // events
    void on_scene_opened(Scene const& scene);
    void on_render_config_change(RenderConfig const& conf);
    void on_mouse_leave_scene_viewport() noexcept;
    void on_mouse_enter_scene_viewport() noexcept;
    void on_obj_change(std::optional<EditableView> editable) noexcept;

    // other helpers
    void try_select_object() noexcept;
    void handle_input(InputEvent const& event) noexcept override;

    void add_object(Object obj) noexcept;
    void add_light(Light light) noexcept;
    void remove_editable(EditableView editable);
    void on_add_editable(EditableView editable);
    void on_editable_change(EditableView editable, EditableChangeType type);
	void on_log_added() override;
    auto get_cur_renderer() noexcept -> Renderer&;

    std::string m_console_text;

    std::function<void()> m_on_mouse_leave_scene_viewport_cb;
    std::function<void()> m_on_mouse_enter_scene_viewport_cb;

    // rendering
    RenderConfig m_config;
    Scene m_scene;
    Camera m_cam;

    // input handling
    std::vector<InputAction> m_input_actions;

    // rendering
    std::vector<std::unique_ptr<Renderer>> m_renderers;
    std::unique_ptr<Archive> m_archive;

    GLTextureRef m_light_icon_tex;

    struct ControlState {
        using ObjChangeCallback = std::function<void(std::optional<EditableView>)>;

        void set_cur_obj(std::optional<EditableView> obj) noexcept;
        auto get_cur_obj() const noexcept { return m_cur_obj; }
        void register_on_obj_change(ObjChangeCallback callback) noexcept;

        float move_sensitivity = k_init_move_sensitivity;
        float rot_sensitivity = k_init_rot_sensitivity;
        std::array<char, 1024> obj_name_buf {};
        bool is_outside_view{ false };
        int cur_renderer_idx{ k_default_renderer_idx };
        std::optional<glm::vec2> last_mouse_pos{ std::nullopt };

        struct GizmoState {
            ImGuizmo::OPERATION op{ ImGuizmo::OPERATION::TRANSLATE };
            ImGuizmo::MODE mode{ ImGuizmo::MODE::WORLD };
            bool snap{ false };
            glm::vec3 snap_scale{ 1.0 };
        } gizmo_state{ };
    private:
        std::optional<EditableView> m_cur_obj;
        std::vector<ObjChangeCallback> m_obj_change_callbacks;
    } m_control_state;
};

template <typename T, typename E>
constexpr decltype(auto) EditorApplication::check_error(tl::expected<T, E> const& res) {
    if(!res) {
        std::cerr << res.error() << std::endl;
        EditorApplication::get().quit(-1);
    }
    if constexpr (!std::is_void_v<T>) {
        return res.value();
    }
}
template <typename T, typename E>
constexpr decltype(auto) EditorApplication::check_error(tl::expected<T, E>&& res) {
    if (!res) {
        std::cerr << res.error() << std::endl;
        EditorApplication::get().quit(-1);
    }
	if constexpr (!std::is_void_v<T>) {
        return std::move(res).value();
    }
}

}
}
