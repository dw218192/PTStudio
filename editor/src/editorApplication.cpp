#include "include/editorApplication.h"

EditorApplication::EditorApplication(Renderer& renderer, std::string_view const& name)
	: Application{ renderer, name } {
    auto scene = check_error(Scene::from_obj_file("C:/Users/tongw/Dropbox/repos/PTStudio/_files/ada.obj"));
    check_error(get_renderer().open_scene(scene));
}

void EditorApplication::cursor_moved(double x, double y) {
    if (cur_button_down == -1) {
        return;
    }

    if (!first_time_motion) {
        auto px = (x - prev_x) / get_window_width();
        auto py = (y - prev_y) / get_window_height();

        Cmd cmd;
        auto move_sensitivity = m_control_state.move_sensitivity;
        auto rot_sensitivity = m_control_state.rot_sensitivity;
        auto zoom_sensitivity = m_control_state.zoom_sensitivity;

        if (cur_button_down == GLFW_MOUSE_BUTTON_LEFT) {
            cmd = Cmd_CameraMove{
                {
                    move_sensitivity * px, -move_sensitivity * py, 0
                }
            };
        }
        else if (cur_button_down == GLFW_MOUSE_BUTTON_RIGHT) {
            cmd = Cmd_CameraRot{
                {
                    rot_sensitivity * py, rot_sensitivity * px, 0
                }
            };
        }
        else if (cur_button_down == GLFW_MOUSE_BUTTON_MIDDLE) {
            cmd = Cmd_CameraZoom{
                static_cast<float>(py * zoom_sensitivity)
            };
        }

        check_error(get_renderer().exec(cmd));
    }
    else {
        first_time_motion = false;
    }
    prev_x = x;
    prev_y = y;
}

void EditorApplication::mouse_clicked(int button, int action, int mods) {
    cur_button_down = button;
    if (action == GLFW_PRESS) {
        first_time_motion = true;
    }
    else if (action == GLFW_RELEASE) {
        cur_button_down = -1;
    }
}

void EditorApplication::mouse_scroll(double x, double y) {
    (void)x;
    float const delta = y < 0 ? -1.0f : 1.0f;
    check_error(get_renderer().exec(Cmd_CameraZoom{ delta }));
}


void EditorApplication::draw_imgui() {
    ImGui::Begin("Renderer");

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Control Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("Camera Control");
        ImGui::SliderFloat("Move Sensitivity", &m_control_state.move_sensitivity, 1.0f, 10.0f);
        ImGui::SliderFloat("Rotate Sensitivity", &m_control_state.rot_sensitivity, 2.0f, 100.0f);
        ImGui::SliderFloat("Zoom Sensitivity", &m_control_state.zoom_sensitivity, 1.0f, 20.0f);
    }

    ImGui::End();
}

void EditorApplication::loop() {
    draw_imgui();

    if (!get_renderer().valid()) {
        return;
    }
    check_error(get_renderer().render());
}