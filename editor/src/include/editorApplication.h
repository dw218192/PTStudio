#pragma once
#include <string>

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
    bool first_time_motion = true;
    int cur_button_down = -1;
    double prev_x = 0, prev_y = 0;
    std::string cur_obj_file;

    void draw_scene_panel() noexcept;
    void draw_object_panel() noexcept;
    void draw_scene_viewport(TextureRef render_buf) noexcept;
    void draw_console_panel() noexcept;

    struct {
        float move_sensitivity = k_init_move_sensitivity;
        float rot_sensitivity = k_init_rot_sensitivity;
        float zoom_sensitivity = k_init_zoom_sensitivity;
    } m_control_state;
};

