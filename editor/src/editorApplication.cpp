#include "include/editorApplication.h"
#include "include/debugDrawer.h"
#include "include/editorResources.h"

EditorApplication::EditorApplication(Renderer& renderer, Scene& scene, std::string_view name)
	: Application{ renderer, scene, name } 
{
// #define EDITOR_APP_IMGUI_LOAD_INI
#ifndef EDITOR_APP_IMGUI_LOAD_INI
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::LoadIniSettingsFromMemory(k_imgui_ini, std::strlen(k_imgui_ini));
#endif
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

void EditorApplication::loop() {
    // create an UI that covers the whole window, for docking
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
    auto render_tex = check_error(get_renderer().render_buffered());

    // draw left panel
    begin_imgui_window("Scene Settings");
    {
        draw_scene_panel();
    }
    end_imgui_window();

    // draw right panel
    begin_imgui_window("Inspector");
    {
        draw_object_panel();
    }
    end_imgui_window();
    
    // draw bottom panel
    begin_imgui_window("Console");
    {
        draw_console_panel();
    }
    end_imgui_window();

    // draw the scene view
    begin_imgui_window("Scene", true, ImGuiWindowFlags_NoScrollWithMouse);
    {
        draw_scene_viewport(render_tex);
    }
    end_imgui_window();
}

void EditorApplication::draw_scene_panel() noexcept {
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Control Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("Camera Control");
        ImGui::SliderFloat("Move Sensitivity", &m_control_state.move_sensitivity, 1.0f, 10.0f);
        ImGui::SliderFloat("Rotate Sensitivity", &m_control_state.rot_sensitivity, 2.0f, 100.0f);
        ImGui::SliderFloat("Zoom Sensitivity", &m_control_state.zoom_sensitivity, 1.0f, 20.0f);
    }

    ImGui::Spacing();
    // draw scene object list
    if (ImGui::CollapsingHeader("Scene Objects", ImGuiTreeNodeFlags_DefaultOpen))
    {
        auto const& objects = get_scene().get_objects();

        for (auto&& obj : objects) {
            // TODO
        }
    }
}

void EditorApplication::draw_object_panel() noexcept {
    ImGui::Spacing();
    ImGui::Text("not implemented");
}

void EditorApplication::draw_scene_viewport(TextureRef render_buf) noexcept {
    if (get_renderer().valid()) {
        static auto last_size = ImVec2{ 0, 0 };
        auto view_size = ImGui::GetContentRegionAvail();

        if (std::abs(view_size.x - last_size.x) >= 0.01f || std::abs(view_size.y - last_size.y) >= 0.01f) {
            auto conf = get_renderer().get_config();
            conf.width = static_cast<unsigned>(view_size.x);
            conf.height = static_cast<unsigned>(view_size.y);

            check_error(get_renderer().exec(Cmd_ChangeRenderConfig{ conf }));
            last_size = view_size;
        }

        try {
            auto id = std::any_cast<GLuint>(render_buf.get().get_handle());

            glBindTexture(GL_TEXTURE_2D, id);
            ImGui::Image(reinterpret_cast<ImTextureID>(id), view_size, { 0, 1 }, { 1, 0 });
            glBindTexture(GL_TEXTURE_2D, 0);

        }
        catch (std::bad_any_cast const& e) {
            std::cerr << e.what() << '\n';
            Application::quit(-1);
        }

        auto pos = ImGui::GetWindowPos();
        DebugDrawer::set_draw_list(ImGui::GetWindowDrawList());

        // draw x,y,z axis ref
        auto axis_origin = get_cam().viewport_to_world(glm::vec2 { pos.x + 30, pos.y + 50 });
        constexpr float axis_len = 0.01f;
        DebugDrawer::draw_line_3d(axis_origin, axis_origin + glm::vec3{ axis_len, 0, 0 }, { 1, 0, 0 });
        DebugDrawer::draw_line_3d(axis_origin, axis_origin + glm::vec3{ 0, axis_len, 0 }, { 0, 1, 0 });
        DebugDrawer::draw_line_3d(axis_origin, axis_origin + glm::vec3{ 0, 0, axis_len }, { 0, 0, 1 });
        DebugDrawer::set_draw_list(nullptr);
    } else {
        ImGui::Text("Renderer not found");
    }
}

void EditorApplication::draw_console_panel() noexcept {
    ImGui::Spacing();
    ImGui::Text("not implemented");
}
