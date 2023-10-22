#pragma once

#include <imgui.h>
#include <ImGuizmo.h>
#include <array>

#include "scene.h"
#include "application.h"
#include "editorConsole.h"

constexpr float k_init_move_sensitivity = 5.0;
constexpr float k_init_rot_sensitivity = 60.0;
constexpr float k_object_select_mouse_time = 1.0f;

struct EditorApplication : Application {
    EditorApplication(Renderer& renderer, Scene& scene, std::string_view name);

    void cursor_moved(double x, double y) override;
    void mouse_clicked(int button, int action, int mods) override;
    void mouse_scroll(double x, double y) override;
    void key_pressed(int key, int scancode, int action, int mods) override;

    void loop(float dt) override;

private:
    // imgui rendering
    void draw_scene_panel() noexcept;
    void draw_object_panel() noexcept;
    void draw_scene_viewport(TextureHandle render_buf) noexcept;
    void draw_console_panel() noexcept;

    // control conditions
    bool can_rotate() const noexcept;
    bool can_move() const noexcept;

    // events
    void on_mouse_leave_scene_viewport() noexcept;
    void on_mouse_enter_scene_viewport() noexcept;
    void on_obj_change(Object* obj) noexcept;

    // other helpers
    void try_select_object() noexcept;
    void handle_key_release() noexcept;
    void handle_mouse_press(int button) noexcept;
    void handle_mouse_release() noexcept;
    void add_object(Object const& obj) noexcept;
    void remove_object(ObjectHandle obj) noexcept;

    std::function<void()> m_on_mouse_leave_scene_viewport_cb;
    std::function<void()> m_on_mouse_enter_scene_viewport_cb;

    EditorConsole<5> m_console;
    
    struct ControlState {
        using ObjChangeCallback = std::function<void(ObjectHandle)>;

        void set_cur_obj(ObjectHandle obj) noexcept;
        auto get_cur_obj() const noexcept { return m_cur_obj; }
        void register_on_obj_change(ObjChangeCallback callback) noexcept;

        float move_sensitivity = k_init_move_sensitivity;
        float rot_sensitivity = k_init_rot_sensitivity;
        std::array<char, 1024> obj_name_buf {};

        bool is_outside_view{ false };

        struct InputState {
            bool first_time_motion{ true };
            int cur_mouse_down{ -1 };
            int cur_button_down{ -1 };
            double mouse_down_time { glfwGetTime() };
            double prev_x{ 0 }, prev_y{ 0 };
        } input_state { };

        struct GizmoState {
            ImGuizmo::OPERATION op{ ImGuizmo::OPERATION::TRANSLATE };
            ImGuizmo::MODE mode{ ImGuizmo::MODE::WORLD };
            bool snap{ false };
            glm::vec3 snap_scale{ 1.0 };
        } gizmo_state{ };
    private:
        ObjectHandle m_cur_obj = nullptr;
        std::vector<ObjChangeCallback> m_obj_change_callbacks;
    } m_control_state;
};

