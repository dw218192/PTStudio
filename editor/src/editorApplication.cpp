#include "include/editorApplication.h"
#include "include/debugDrawer.h"
#include "include/editorResources.h"
#include "include/imgui/editorFields.h"
#include "include/imgui/fileDialogue.h"
#include "include/imgui/imhelper.h"
#include "include/editorConsole.h"

EditorApplication::EditorApplication(Renderer& renderer, Scene& scene, std::string_view name)
	: Application{ renderer, scene, name } 
{
// #define EDITOR_APP_IMGUI_LOAD_INI
#ifndef EDITOR_APP_IMGUI_LOAD_INI
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::LoadIniSettingsFromMemory(k_imgui_ini, std::strlen(k_imgui_ini));
#endif

    m_control_state.register_on_obj_change(&EditorApplication::on_obj_change);
    m_on_mouse_leave_scene_viewport_cb = [this] { on_mouse_leave_scene_viewport(); };
}

void EditorApplication::cursor_moved(double x, double y) {
    if (m_control_state.cur_mouse_down == -1) {
        return;
    }

    if (!m_control_state.first_time_motion) {
        auto px = (x - m_control_state.prev_x) / get_window_width();
        auto py = (y - m_control_state.prev_y) / get_window_height();

        auto move_sensitivity = m_control_state.move_sensitivity;
        auto rot_sensitivity = m_control_state.rot_sensitivity;
        auto zoom_sensitivity = m_control_state.zoom_sensitivity;

        if (can_move()) {
            get_cam().set_delta_dolly({
                move_sensitivity * px, 0, move_sensitivity* py
            });
        } else if (can_rotate()) {
            get_cam().set_delta_rotation({
                rot_sensitivity * py, rot_sensitivity * px, 0
            });
        }
    }
    else {
        m_control_state.first_time_motion = false;
    }
    m_control_state.prev_x = x;
    m_control_state.prev_y = y;
}

void EditorApplication::mouse_clicked(int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        m_control_state.cur_mouse_down = button;
        m_control_state.first_time_motion = true;
    } else if (action == GLFW_RELEASE) {
        m_control_state.cur_mouse_down = -1;
        try_select_object();
    }
}

void EditorApplication::mouse_scroll(double x, double y) {
    (void)x;
    float const delta = y < 0 ? -1.0f : 1.0f;
    get_cam().set_delta_zoom(delta);
}

void EditorApplication::key_pressed(int key, int scancode, int action, int mods) {
    (void)scancode;
    (void)mods;
    if (action == GLFW_PRESS) {
        m_control_state.cur_button_down = key;
    } else if (action == GLFW_RELEASE) {
        handle_key_release();
        m_control_state.cur_button_down = -1;
    }
}

void EditorApplication::loop(float dt) {
    // create an UI that covers the whole window, for docking
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
    auto render_tex = check_error(get_renderer().render_buffered(get_cam()));

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
    begin_imgui_window("Scene", true, ImGuiWindowFlags_NoScrollWithMouse, m_on_mouse_leave_scene_viewport_cb);
    {
        draw_scene_viewport(render_tex);
    }
    end_imgui_window();
}

void EditorApplication::draw_scene_panel() noexcept {
    if (ImGui::CollapsingHeader("Camera Control", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("Move Sensitivity");
        ImGui::SliderFloat("##Move Sensitivity", &m_control_state.move_sensitivity, 1.0f, 10.0f);
        ImGui::Text("Rotate Sensitivity");
        ImGui::SliderFloat("##Rotate Sensitivity", &m_control_state.rot_sensitivity, 2.0f, 100.0f);
        ImGui::Text("Zoom Sensitivity");
        ImGui::SliderFloat("##Zoom Sensitivity", &m_control_state.zoom_sensitivity, 1.0f, 20.0f);
    }

    // draw scene object list
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Scene Objects", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::BeginListBox("##Scene Objects", { 0, 200 });
        {
            for (auto obj : get_scene()) {
                if (ImGui::Selectable(obj->get_name().data(), m_control_state.get_cur_obj() == obj)) {
                    m_control_state.set_cur_obj(obj);
                }
            }
        }
        ImGui::EndListBox();
        ImGui::Spacing();

        if (ImGui::BeginMenu("Add Object")) {
            if (ImGui::MenuItem("Triangle")) {
                add_object(Object::make_triangle_obj(get_scene(), Material{}, Transform{}));
            }
            if (ImGui::MenuItem("Cube")) {

            }
            if (ImGui::MenuItem("Sphere")) {

            }
            if (ImGui::MenuItem("Import .obj File")) {
                auto path = ImGui::FileDialogue("obj");
                if(!path.empty()) {
                    std::string warning;
                    auto obj = check_error(Object::from_obj(get_scene(), Material{}, path, &warning));
                    add_object(obj);

                    m_console.log(warning);
                }
            }
            ImGui::EndMenu();
        }
    }
}

void EditorApplication::draw_object_panel() noexcept {
    if (ImGui::CollapsingHeader("Object Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (m_control_state.get_cur_obj()) {
            auto& obj = *m_control_state.get_cur_obj();

            // NOTE: no safety check here, cuz I'm lazy
            if (ImGui::InputText("Name", m_control_state.obj_name_buf.data(), m_control_state.obj_name_buf.size(),
                ImGuiInputTextFlags_EnterReturnsTrue)) 
            {
                obj.set_name(m_control_state.obj_name_buf.data());
            }
            auto trans = obj.get_transform();
            if (ImGui::TransformField("Transform", trans)) {
                obj.set_transform(trans);
            }
        } else {
            ImGui::Text("No object selected");
        }
    }
}

void EditorApplication::draw_scene_viewport(TextureRef render_buf) noexcept {
    if (get_renderer().valid()) {
        static auto last_size = ImVec2{ 0, 0 };

        auto v_min = ImGui::GetWindowContentRegionMin();
        auto v_max = ImGui::GetWindowContentRegionMax();
        auto view_size = v_max - v_min;

        if (std::abs(view_size.x - last_size.x) >= 0.01f || std::abs(view_size.y - last_size.y) >= 0.01f) {
            auto conf = get_renderer().get_config();
            conf.width = static_cast<unsigned>(view_size.x);
            conf.height = static_cast<unsigned>(view_size.y);
            get_cam().set_viewport(conf.width, conf.height);
            get_cam().set_fov(conf.fovy);
            check_error(get_renderer().on_change_render_config(conf));
            last_size = view_size;
        }

        try {
            render_buf.get().bind();
            ImGui::Image(
	            reinterpret_cast<ImTextureID>(std::any_cast<GLuint>(render_buf.get().get_handle())),
                view_size, { 0, 1 }, { 1, 0 }
            );
            render_buf.get().unbind();
        }
        catch (std::bad_any_cast const& e) {
            std::cerr << e.what() << '\n';
            Application::quit(-1);
        }

        // draw x,y,z axis ref
        auto axis_origin = get_cam().viewport_to_world(glm::vec2 {30, 30}, 0.0f);
        constexpr float axis_len = 0.01f;

        get_debug_drawer().begin_relative(to_glm(ImGui::GetWindowPos() + v_min));
        get_debug_drawer().draw_line_3d(axis_origin, axis_origin + glm::vec3{ axis_len, 0, 0 }, { 1, 0, 0 }, 2.0f, 0.0f);
        get_debug_drawer().draw_line_3d(axis_origin, axis_origin + glm::vec3{ 0, axis_len, 0 }, { 0, 1, 0 }, 2.0f, 0.0f);
        get_debug_drawer().draw_line_3d(axis_origin, axis_origin + glm::vec3{ 0, 0, axis_len }, { 0, 0, 1 }, 2.0f, 0.0f);
        get_debug_drawer().end_relative();
    } else {
        ImGui::Text("Renderer not found");
    }
}

void EditorApplication::draw_console_panel() noexcept {
    ImGui::BeginChild("##scroll");
    {
        ImGui::TextUnformatted(m_console.str().data());
    }
    ImGui::EndChild();
}

bool EditorApplication::can_rotate() const noexcept {
    return m_control_state.cur_mouse_down == GLFW_MOUSE_BUTTON_RIGHT;
}

bool EditorApplication::can_move() const noexcept {
    return m_control_state.cur_mouse_down == GLFW_MOUSE_BUTTON_LEFT;
}

void EditorApplication::on_mouse_leave_scene_viewport() noexcept {
    m_control_state.cur_mouse_down = -1;
    m_control_state.cur_button_down = -1;
    m_control_state.first_time_motion = true;
}

void EditorApplication::on_obj_change(Object* obj) noexcept { }

void EditorApplication::try_select_object() noexcept {
    if (get_cur_hovered_widget() != "Scene") {
        return;
    }
    auto pos = ImGui::GetMousePos();
    auto win_pos = get_window_content_pos("Scene");
    if (!win_pos) {
        std::cerr << "Scene view not found" << std::endl;
        return;
    }

    // convert to viewport space
    pos = pos - win_pos.value();
    auto ray = get_cam().viewport_to_ray(to_glm(pos));
    auto res = get_scene().ray_cast(ray);
    m_control_state.set_cur_obj(res);

    // m_console.log("Selected object: ", res ? res->get_name() : "None");
}

void EditorApplication::handle_key_release() noexcept {
    if (m_control_state.cur_button_down == GLFW_KEY_DELETE) {
        if (m_control_state.get_cur_obj()) {
            remove_object(m_control_state.get_cur_obj());
        }
    }
}

void EditorApplication::add_object(Object const& obj) noexcept {
    auto hobj = get_scene().add_object(obj);
    check_error(get_renderer().on_add_object(hobj));
    m_control_state.set_cur_obj(hobj);
}

void EditorApplication::remove_object(ObjectHandle obj) noexcept {
    check_error(get_renderer().on_remove_object(obj));
    get_scene().remove_object(m_control_state.get_cur_obj());
	m_control_state.set_cur_obj(nullptr);
}

void EditorApplication::ControlState::set_cur_obj(ObjectHandle obj) noexcept {
    if (obj == m_cur_obj) {
        return;
    }
    m_cur_obj = obj;
    for (auto&& callback : m_obj_change_callbacks) {
        (dynamic_cast<EditorApplication*>(&Application::get_application())->*callback)(obj);
    }
    if (obj) {
        std::copy(obj->get_name().begin(), obj->get_name().end(), obj_name_buf.begin());
    }
}

void EditorApplication::ControlState::register_on_obj_change(ObjChangeCallback callback) noexcept {
    m_obj_change_callbacks.emplace_back(callback);
}
