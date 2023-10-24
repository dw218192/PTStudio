#include "include/editorApplication.h"
#include "include/debugDrawer.h"
#include "include/editorResources.h"
#include "include/imgui/editorFields.h"
#include "include/imgui/fileDialogue.h"
#include "include/imgui/imhelper.h"
#include "include/editorRenderer.h"

#include <imgui_internal.h>

static constexpr char const* k_scene_view_win_name = "Scene";

EditorApplication::EditorApplication(Renderer& renderer, Scene& scene, std::string_view name)
    : GLFWApplication { name, renderer.get_config().width, renderer.get_config().height, renderer.get_config().min_frame_time },
	  m_scene { scene }, m_renderer{ renderer },
      m_cam{ renderer.get_config().fovy, renderer.get_config().width, renderer.get_config().height, scene.get_good_cam_start() }
{
    // initialize renderer
    check_error(m_renderer.init());
	check_error(m_renderer.open_scene(scene));

// #define EDITOR_APP_IMGUI_LOAD_INI
#ifndef EDITOR_APP_IMGUI_LOAD_INI
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::LoadIniSettingsFromMemory(k_imgui_ini, std::strlen(k_imgui_ini));
#endif

    m_control_state.register_on_obj_change([this] (ObserverPtr<Object> obj) {
	    on_obj_change(obj);
    });
    if(auto p_editor_renderer = dynamic_cast<EditorRenderer*>(&renderer); p_editor_renderer) {
        m_control_state.register_on_obj_change([p_editor_renderer](ObserverPtr<Object> obj) {
            p_editor_renderer->on_object_change(obj);
        });
    }
    m_on_mouse_leave_scene_viewport_cb = [this] { on_mouse_leave_scene_viewport(); };
    m_on_mouse_enter_scene_viewport_cb = [this] { on_mouse_enter_scene_viewport(); };
}

void EditorApplication::cursor_moved(double x, double y) {
    auto& input_state = m_control_state.input_state;
    if (input_state.cur_mouse_down == -1) {
        return;
    }

    if (!input_state.first_time_motion) {
        auto const px = (x - input_state.prev_x) / get_window_width();
        auto const py = (y - input_state.prev_y) / get_window_height();

        auto const move_sensitivity = m_control_state.move_sensitivity;
        auto const rot_sensitivity = m_control_state.rot_sensitivity;

        if (can_move()) {
            m_cam.set_delta_dolly({
                move_sensitivity * px, 0, move_sensitivity* py
            });
        } else if (can_rotate()) {
            m_cam.set_delta_rotation({
                rot_sensitivity * py, rot_sensitivity * px, 0
            });
        }
    }
    else {
        input_state.first_time_motion = false;
    }
    input_state.prev_x = x;
    input_state.prev_y = y;
}

void EditorApplication::mouse_clicked(int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        handle_mouse_press(button);
    } else if (action == GLFW_RELEASE) {
        handle_mouse_release();
    }
}

void EditorApplication::mouse_scroll(double x, double y) {
    (void)x;
    float const delta = y < 0 ? -1.0f : 1.0f;
    m_cam.set_delta_zoom(delta);
}

void EditorApplication::key_pressed(int key, int scancode, int action, int mods) {
    (void)scancode;
    (void)mods;

    auto& input_state = m_control_state.input_state;
    if (action == GLFW_PRESS) {
        input_state.cur_button_down = key;
    } else if (action == GLFW_RELEASE) {
        handle_key_release();
    }
}

void EditorApplication::loop(float dt) {
    // ImGuizmo
    ImGuizmo::BeginFrame();

    // create an UI that covers the whole window, for docking
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
    auto render_tex = check_error(m_renderer.render_buffered(m_cam));

    // draw left panel
    begin_imgui_window("Scene Settings", ImGuiWindowFlags_NoMove);
    {
        draw_scene_panel();
    }
    end_imgui_window();

    // draw right panel
    begin_imgui_window("Inspector", ImGuiWindowFlags_NoMove);
    {
        draw_object_panel();
    }
    end_imgui_window();
    
    // draw bottom panel
    begin_imgui_window("Console", ImGuiWindowFlags_NoMove);
    {
        draw_console_panel();
    }
    end_imgui_window();

    // draw the scene view
    begin_imgui_window(k_scene_view_win_name,
        ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoMove,
        m_on_mouse_leave_scene_viewport_cb,
        m_on_mouse_enter_scene_viewport_cb);
    {
        draw_scene_viewport(render_tex);
    }
    end_imgui_window();

    check_error(m_renderer.draw_imgui(this));
}

void EditorApplication::quit(int code) {
    get().~EditorApplication();
	std::exit(code);
}

void EditorApplication::draw_scene_panel() noexcept {
    if (ImGui::CollapsingHeader("Camera Control", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("Move Sensitivity");
        ImGui::SliderFloat("##Move Sensitivity", &m_control_state.move_sensitivity, 1.0f, 10.0f);
        ImGui::Text("Rotate Sensitivity");
        ImGui::SliderFloat("##Rotate Sensitivity", &m_control_state.rot_sensitivity, 2.0f, 100.0f);
    }

    // draw scene object list
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Scene Objects", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::BeginListBox("##Scene Objects", { 0, 200 }))
        {
            for (auto obj : m_scene) {
                if (ImGui::Selectable(obj->get_name().data(), m_control_state.get_cur_obj() == obj)) {
                    m_control_state.set_cur_obj(obj);
                }
            }

            ImGui::EndListBox();
        }
        ImGui::Spacing();

        if (ImGui::BeginMenu("Add Object")) {
            if (ImGui::MenuItem("Triangle")) {
                add_object(Object::make_triangle_obj(m_scene, Material{}, Transform{}));
            }
            if (ImGui::MenuItem("Cube")) {

            }
            if (ImGui::MenuItem("Sphere")) {

            }
            if (ImGui::MenuItem("Import .obj File")) {
                auto path = ImGui::FileDialogue("obj");
                if(!path.empty()) {
                    std::string warning;
                    auto obj = check_error(Object::from_obj(m_scene, Material{}, path, &warning));
                    add_object(obj);

                    this->log(warning);
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

            auto&& gizmo_state = m_control_state.gizmo_state;
            auto trans = obj.get_transform();
            if (ImGui::TransformField("Transform", trans, gizmo_state.op, gizmo_state.mode, gizmo_state.snap, gizmo_state.snap_scale)) {
                obj.set_transform(trans);
            }
        } else {
            ImGui::Text("No object selected");
        }
    }
}

void EditorApplication::draw_scene_viewport(TextureHandle render_buf) noexcept {
    if (m_renderer.valid()) {
        static auto last_size = ImVec2{ 0, 0 };

        auto const v_min = ImGui::GetWindowContentRegionMin();
        auto const v_max = ImGui::GetWindowContentRegionMax();
        auto const view_size = v_max - v_min;

        if (std::abs(view_size.x - last_size.x) >= 0.01f || std::abs(view_size.y - last_size.y) >= 0.01f) {
            auto conf = m_renderer.get_config();
            conf.width = static_cast<unsigned>(view_size.x);
            conf.height = static_cast<unsigned>(view_size.y);
            m_cam.set_viewport(conf.width, conf.height);
            m_cam.set_fov(conf.fovy);
            check_error(m_renderer.on_change_render_config(conf));
            last_size = view_size;
        }

        check_error(render_buf->bind());
        ImGui::Image(
            render_buf->get_id(),
            view_size, { 0, 1 }, { 1, 0 }
        );
        render_buf->unbind();

        // draw x,y,z axis ref
        auto const axis_origin = m_cam.viewport_to_world(glm::vec2 {30, 30}, 0.0f);
        constexpr float axis_len = 0.01f;

        get_debug_drawer().begin_relative(to_glm(ImGui::GetWindowPos() + v_min));
        get_debug_drawer().draw_line_3d(m_cam, axis_origin, axis_origin + glm::vec3{ axis_len, 0, 0 }, { 1, 0, 0 }, 2.0f, 0.0f);
        get_debug_drawer().draw_line_3d(m_cam, axis_origin, axis_origin + glm::vec3{ 0, axis_len, 0 }, { 0, 1, 0 }, 2.0f, 0.0f);
        get_debug_drawer().draw_line_3d(m_cam, axis_origin, axis_origin + glm::vec3{ 0, 0, axis_len }, { 0, 0, 1 }, 2.0f, 0.0f);
        get_debug_drawer().end_relative();

        // draw gizmos
        if (m_control_state.get_cur_obj()) {
            auto const win = ImGui::GetCurrentWindow();
            auto const& gizmo_state = m_control_state.gizmo_state;

            ImGuizmo::SetDrawlist(win->DrawList);
            ImGuizmo::SetRect(win->Pos.x, win->Pos.y, win->Size.x, win->Size.y);

            auto mat = m_control_state.get_cur_obj()->get_transform().get_matrix();
            if (ImGuizmo::Manipulate(
                glm::value_ptr(m_cam.get_view()),
                glm::value_ptr(m_cam.get_projection()),
                gizmo_state.op,
                gizmo_state.mode,
                glm::value_ptr(mat),
                nullptr,
                gizmo_state.snap ? glm::value_ptr(gizmo_state.snap_scale) : nullptr
            )) {
                m_control_state.get_cur_obj()->set_transform(Transform{ mat });
            }
        }
    } else {
        ImGui::Text("Renderer not found");
    }
}

void EditorApplication::draw_console_panel() const noexcept {
    ImGui::BeginChild("##scroll");
    {
        ImGui::TextUnformatted(m_console_text.data());
    }
    ImGui::EndChild();
}

bool EditorApplication::can_rotate() const noexcept {
    return
        get_cur_focused_widget() == k_scene_view_win_name &&
        !ImGuizmo::IsUsing() &&
        m_control_state.input_state.cur_mouse_down == GLFW_MOUSE_BUTTON_RIGHT;
}

bool EditorApplication::can_move() const noexcept {
    return
        get_cur_focused_widget() == k_scene_view_win_name &&
        !ImGuizmo::IsUsing() &&
        m_control_state.input_state.cur_mouse_down == GLFW_MOUSE_BUTTON_LEFT;
}

void EditorApplication::on_mouse_leave_scene_viewport() noexcept {
    m_control_state.is_outside_view = true;
}

void EditorApplication::on_mouse_enter_scene_viewport() noexcept {
    m_control_state.is_outside_view = false;
}

void EditorApplication::on_obj_change(ObserverPtr<Object> obj) noexcept {}

void EditorApplication::try_select_object() noexcept {
    auto pos = ImGui::GetMousePos();
    auto const win_pos = get_window_content_pos(k_scene_view_win_name);
    if (!win_pos) {
        this->log("scene view not found");
        return;
    } else if(get_cur_hovered_widget() != k_scene_view_win_name) {
        return;
    } else if(get_cur_focused_widget() != k_scene_view_win_name) {
        return;
    }

    // convert to viewport space
    pos = pos - win_pos.value();
    auto const ray = m_cam.viewport_to_ray(to_glm(pos));
    auto const res = m_scene.ray_cast(ray);

    // prevent deselecting when not in scene view
    m_control_state.set_cur_obj(res);

    // m_console.log("Selected object: ", res ? res->get_name() : "None");
}

void EditorApplication::handle_key_release() noexcept {
    auto& gizmo_state = m_control_state.gizmo_state;
    auto& input_state = m_control_state.input_state;
    if (!m_control_state.get_cur_obj() || ImGuizmo::IsUsing()) {
        input_state.cur_button_down = -1;
        return;
    }
    switch(input_state.cur_button_down) {
    case GLFW_KEY_DELETE:
        remove_object(m_control_state.get_cur_obj());
        break;
    case GLFW_KEY_W:
        gizmo_state.op = ImGuizmo::TRANSLATE;
        break;
    case GLFW_KEY_E:
        gizmo_state.op = ImGuizmo::ROTATE;
        break;
    case GLFW_KEY_R:
        gizmo_state.op = ImGuizmo::SCALE;
        break;
    case GLFW_KEY_X:
        gizmo_state.snap = !gizmo_state.snap;
        break;
    default:
        break;
    }

    input_state.cur_button_down = -1;
}

void EditorApplication::handle_mouse_press(int button) noexcept {
    auto& input_state = m_control_state.input_state;
    if (!m_control_state.is_outside_view) {
        // only initiate mouse input in the scene view window
        input_state.mouse_down_time = glfwGetTime();
        input_state.cur_mouse_down = button;
        input_state.first_time_motion = true;
    }
}

void EditorApplication::handle_mouse_release() noexcept {
    auto& input_state = m_control_state.input_state;
    if (!m_control_state.is_outside_view && (glfwGetTime() - input_state.mouse_down_time) <= k_object_select_mouse_time) {
        try_select_object();
    }
    input_state = {};
}

void EditorApplication::add_object(Object const& obj) noexcept {
    auto hobj = m_scene.add_object(obj);
    check_error(m_renderer.on_add_object(hobj));
    m_control_state.set_cur_obj(hobj);
}

void EditorApplication::remove_object(ObserverPtr<Object> obj) noexcept {
    check_error(m_renderer.on_remove_object(obj));
    m_scene.remove_object(m_control_state.get_cur_obj());
	m_control_state.set_cur_obj(nullptr);
}

void EditorApplication::print(std::string_view msg) {
    m_console_text = msg;
}

void EditorApplication::ControlState::set_cur_obj(ObserverPtr<Object> obj) noexcept {
    if (obj == m_cur_obj) {
        return;
    }

    if (ImGuizmo::IsUsing()) {
        // let gizmos finish
    	return;
    }

    m_cur_obj = obj;
    for (auto&& callback : m_obj_change_callbacks) {
        callback(obj);
    }
    if (obj) {
        std::copy(obj->get_name().begin(), obj->get_name().end(), obj_name_buf.begin());
    }
}

void EditorApplication::ControlState::register_on_obj_change(ObjChangeCallback callback) noexcept {
    m_obj_change_callbacks.emplace_back(callback);
}
