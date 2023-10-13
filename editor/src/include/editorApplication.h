#pragma once

#include <string>
#include <array>

#include "scene.h"
#include "application.h"

constexpr float k_init_move_sensitivity = 2.0;
constexpr float k_init_rot_sensitivity = 40.0;
constexpr float k_init_zoom_sensitivity = 10.0;

struct EditorApplication : Application {
    EditorApplication(Renderer& config, Scene& scene, std::string_view name);

    void cursor_moved(double x, double y) override;
    void mouse_clicked(int button, int action, int mods) override;
    void mouse_scroll(double x, double y) override;
    void loop() override;

private:
    void draw_scene_panel() noexcept;
    void draw_object_panel() noexcept;
    void draw_scene_viewport(TextureRef render_buf) noexcept;
    void draw_console_panel() noexcept;

    // events
    void on_mouse_leave_scene_viewport() noexcept;
    void on_obj_change(Object* obj) noexcept;

    std::function<void()> m_on_mouse_leave_scene_viewport_cb;
    
    struct ControlState {
        using ObjChangeCallback = void(EditorApplication::*)(Object*);

        void set_cur_obj(Object* obj) noexcept;
        auto get_cur_obj() const noexcept -> Object* { return m_cur_obj; }
        void register_on_obj_change(ObjChangeCallback callback) noexcept;

        float move_sensitivity = k_init_move_sensitivity;
        float rot_sensitivity = k_init_rot_sensitivity;
        float zoom_sensitivity = k_init_zoom_sensitivity;
        std::array<char, 1024> obj_name_buf {};

        bool first_time_motion = true;
        int cur_button_down = -1;
        double prev_x = 0, prev_y = 0;
    private:
        Object* m_cur_obj = nullptr;
        std::vector<ObjChangeCallback> m_obj_change_callbacks;
    } m_control_state;
};

