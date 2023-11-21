#include "include/editorApplication.h"
#include "include/debugDrawer.h"
#include "include/editorResources.h"
#include "include/imgui/editorFields.h"
#include "include/imgui/fileDialogue.h"
#include "include/imgui/imhelper.h"
#include "include/editorRenderer.h"

#include "vulkanRayTracingRenderer.h"
#include "boundingVolume.h"

#include <filesystem>
#include <imgui_internal.h>

#include "intersection.h"
#include "jsonArchive.h"

using namespace PTS;
using namespace PTS::Editor;

static constexpr auto k_scene_setting_win_name = "Scene Settings";
static constexpr auto k_inspector_win_name = "Inspector";
static constexpr auto k_scene_view_win_name = "Scene";
static constexpr auto k_console_win_name = "Console";

EditorApplication::EditorApplication(std::string_view name, RenderConfig config)
    : GLFWApplication { name, config.width, config.height, config.min_frame_time },
    m_config{config}, m_cam{ config.fovy, config.get_aspect(), LookAtParams{} },
    m_archive{ new JsonArchive }
{

    // initialize gizmo icon textures
    m_light_icon_tex = check_error(GLTexture::create(light_icon_png_data, FileFormat::PNG));

    m_control_state.register_on_obj_change([this] (auto&& obj) {
	    on_obj_change(obj);
    });

    m_on_mouse_leave_scene_viewport_cb = [this] { on_mouse_leave_scene_viewport(); };
    m_on_mouse_enter_scene_viewport_cb = [this] { on_mouse_enter_scene_viewport(); };
    add_renderer(std::make_unique<EditorRenderer>(config));
    add_renderer(std::make_unique<VulkanRayTracingRenderer>(config));
    create_input_actions();
}

void EditorApplication::create_input_actions() noexcept {
    // initialize input actions
    auto obj_selected = InputActionConstraint { [this] (InputEvent const&) {
        return m_control_state.get_cur_obj() && m_control_state.get_cur_obj().has_value();
    }};
    auto using_gizmo = InputActionConstraint { [this] (InputEvent const&) {
        return ImGuizmo::IsUsing();
    }};
    auto not_using_gizmo = InputActionConstraint { [this] (InputEvent const&) {
        return !ImGuizmo::IsUsing();
    }};
    auto scene_view_focused = InputActionConstraint { [this] (InputEvent const&) {
        return get_cur_focused_widget() == k_scene_view_win_name;
    }};
    auto scene_view_hovered = InputActionConstraint { [this] (InputEvent const&) {
        return get_cur_hovered_widget() == k_scene_view_win_name;
    }};
    auto initiated_in_scene_view = InputActionConstraint { [this] (InputEvent const& event) {
        return event.initiated_window == k_scene_view_win_name;
    }};

    // object manipulation

    auto translate_mode = InputAction {{ InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_W },
        [this] (InputEvent const& event) {
            m_control_state.gizmo_state.op = ImGuizmo::OPERATION::TRANSLATE;
        }
    };
    auto rotate_mode = InputAction {{ InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_E },
        [this] (InputEvent const& event) {
            m_control_state.gizmo_state.op = ImGuizmo::OPERATION::ROTATE;
        }
    };
    auto scale_mode = InputAction {{InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_R },
        [this] (InputEvent const& event) {
            m_control_state.gizmo_state.op = ImGuizmo::OPERATION::SCALE;
        }
    };
    auto toggle_snap = InputAction {{InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_X },
        [this] (InputEvent const& event) {
            m_control_state.gizmo_state.snap = !m_control_state.gizmo_state.snap;
        }
    };

    auto delete_obj = InputAction {{ InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_Delete },
        [this] (InputEvent const& event) {
            remove_editable(m_control_state.get_cur_obj().value());
        }
    };
    auto focus_on_obj = InputAction {{InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_F },
        [this] (InputEvent const& event) {
            BoundingBox local_bound;
            if (auto obj = m_control_state.get_cur_obj().value().as<Object>()) {
                local_bound = obj->get_bound();
            } else if (auto light = m_control_state.get_cur_obj().value().as<Light>()) {
                local_bound = BoundingBox{ glm::vec3 { -0.5f }, glm::vec3 { 0.5f } };
            }
            LookAtParams params;
            params.center = m_control_state.get_cur_obj()->get_transform().get_position();
            params.eye = params.center + local_bound.get_extent() * 2.0f;
            params.up = glm::vec3{ 0, 1, 0 };
            m_cam.set(params);
        }
    };
    auto select_obj = InputAction {{InputType::MOUSE, ActionType::RELEASE, ImGuiMouseButton_Left },
        [this] (InputEvent const& event) { 
            if (!m_control_state.is_outside_view && 
                (glfwGetTime() - event.time) <= k_object_select_mouse_time) {
                try_select_object();
            }
        }
    };
    auto deselect_obj = InputAction {{InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_Escape },
        [this] (InputEvent const& event) {
            m_control_state.set_cur_obj(std::nullopt);
        }
    };

    auto cam_pedestal = InputAction {{InputType::MOUSE, ActionType::HOLD, ImGuiMouseButton_Middle}, 
        [this] (InputEvent const& event) {
            m_cam.set_delta_dolly({
                0, m_control_state.move_sensitivity * event.normalized_mouse_delta.y, 0
            });
        }
    };
    auto cam_pan = InputAction {{InputType::MOUSE, ActionType::HOLD, ImGuiMouseButton_Left },
        [this] (InputEvent const& event) {
            m_cam.set_delta_dolly({
                -m_control_state.move_sensitivity * event.normalized_mouse_delta.x,
                0,
                m_control_state.move_sensitivity * event.normalized_mouse_delta.y
            });
        }
    };
    auto cam_rotate = InputAction {{InputType::MOUSE, ActionType::HOLD, ImGuiMouseButton_Right},
        [this] (InputEvent const& event) {
            m_cam.set_delta_rotation({
                m_control_state.rot_sensitivity * event.normalized_mouse_delta.y,
                m_control_state.rot_sensitivity * event.normalized_mouse_delta.x,
                0
            });
        }
    }; 
    auto cam_zoom = InputAction {{InputType::MOUSE, ActionType::SCROLL, ImGuiMouseButton_Middle },
        [this] (InputEvent const& event) {
            m_cam.set_delta_zoom(event.mouse_scroll_delta.y);
        }
    };

    m_input_actions = {
        translate_mode
            .add_constraint(scene_view_focused)
            .add_constraint(scene_view_hovered),
        rotate_mode
            .add_constraint(scene_view_focused)
            .add_constraint(scene_view_hovered),
        scale_mode
            .add_constraint(scene_view_focused)
            .add_constraint(scene_view_hovered),
        toggle_snap
            .add_constraint(scene_view_focused)
            .add_constraint(scene_view_hovered)
            .add_constraint(using_gizmo),
        delete_obj
            .add_constraint(scene_view_focused)
            .add_constraint(scene_view_hovered)
            .add_constraint(obj_selected),
        focus_on_obj
            .add_constraint(obj_selected),
        select_obj
            .add_constraint(scene_view_focused)
            .add_constraint(scene_view_hovered),
        deselect_obj
            .add_constraint(scene_view_focused)
            .add_constraint(scene_view_hovered)
            .add_constraint(obj_selected),
        cam_pedestal
            .add_constraint(initiated_in_scene_view)
            .add_constraint(not_using_gizmo),
        cam_pan
            .add_constraint(initiated_in_scene_view)
            .add_constraint(not_using_gizmo),
        cam_rotate
            .add_constraint(initiated_in_scene_view)
            .add_constraint(not_using_gizmo),
        cam_zoom
            .add_constraint(scene_view_focused)
            .add_constraint(scene_view_hovered)
            .add_constraint(not_using_gizmo),
    };
}

void EditorApplication::add_renderer(std::unique_ptr<Renderer> renderer) noexcept {
    if (!renderer) {
        this->log(LogLevel::Error, "add_renderer(): renderer is null");
        return;
    }
    
    // initialize renderer
    check_error(renderer->init(this));
    check_error(renderer->open_scene(m_scene));

    // editor renderer-specific initialization
    if(auto p_editor_renderer = dynamic_cast<EditorRenderer*>(renderer.get()); p_editor_renderer) {
        m_control_state.register_on_obj_change([p_editor_renderer](auto&& editable) {
            p_editor_renderer->on_selected_editable_change(editable);
        });
    }

    m_renderers.emplace_back(std::move(renderer));
}

void EditorApplication::on_begin_first_loop() {
    GLFWApplication::on_begin_first_loop();

	// do layout initialization work
    if (ImGui::GetIO().IniFilename) {
	    // if we already have a saved layout, do nothing
        if (std::filesystem::exists(ImGui::GetIO().IniFilename)) {
            return;
        }
    }

	// create an UI that covers the whole window, for docking
    auto id = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::DockBuilderRemoveNode(id);
    ImGui::DockBuilderAddNode(id);

    auto const left = ImGui::DockBuilderSplitNode(id, ImGuiDir_Left, 0.146f, nullptr, &id);
    auto const right = ImGui::DockBuilderSplitNode(id, ImGuiDir_Right, 0.160f, nullptr, &id);
    auto const down = ImGui::DockBuilderSplitNode(id, ImGuiDir_Down, 0.245f, nullptr, &id);

    ImGui::DockBuilderDockWindow(k_scene_setting_win_name, left);
    ImGui::DockBuilderDockWindow(k_scene_view_win_name, id);
    ImGui::DockBuilderDockWindow(k_inspector_win_name, right);
    ImGui::DockBuilderDockWindow(k_console_win_name, down);
}

void EditorApplication::loop(float dt) {
    // ImGuizmo
    ImGuizmo::BeginFrame();

    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
    auto const render_tex = check_error(get_cur_renderer().render_buffered(m_cam));

    // draw left panel
    if (begin_imgui_window(k_scene_setting_win_name, ImGuiWindowFlags_NoMove))
    {
        draw_scene_panel();
    }
    end_imgui_window();

    // draw right panel
    if (begin_imgui_window(k_inspector_win_name, ImGuiWindowFlags_NoMove))
    {
        draw_object_panel();
    }
    end_imgui_window();
    
    // draw bottom panel
    if (begin_imgui_window(k_console_win_name, ImGuiWindowFlags_NoMove))
    {
        draw_console_panel();
    }
    end_imgui_window();

    // draw the scene view
    if (begin_imgui_window(k_scene_view_win_name,
        ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_MenuBar,
        m_on_mouse_leave_scene_viewport_cb,
        m_on_mouse_enter_scene_viewport_cb))
    {
        draw_scene_viewport(render_tex);
    }
    end_imgui_window();

    check_error(get_cur_renderer().draw_imgui());
}

void EditorApplication::quit(int code) {
	std::exit(code);
}

void EditorApplication::draw_scene_panel() noexcept {
    ImGui::TextUnformatted(k_editor_tutorial_text);
    ImGui::Separator();

    if (ImGui::Button("Open Scene")) {
        auto const path = ImGui::FileDialogue(ImGui::FileDialogueMode::OPEN, m_archive->get_ext().data());
        if (!path.empty()) {
            std::tie(m_scene, m_cam) = check_error(m_archive->load_file(path));
            on_scene_opened(m_scene);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Save Scene")) {
        auto const path = ImGui::FileDialogue(ImGui::FileDialogueMode::SAVE, m_archive->get_ext().data());
        if (!path.empty()) {
            check_error(m_archive->save_file(m_scene, m_cam, path));
        }
    }

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
            for (auto&& editable : m_scene.get_editables()) {
                if (ImGui::Selectable(editable.get_name().data(), m_control_state.get_cur_obj() == editable)) {
                    m_control_state.set_cur_obj(editable);
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
                add_object(Object::make_cube_obj(m_scene, Material{}, Transform{}));
            }
            if (ImGui::MenuItem("Sphere")) {
                add_object(Object::make_sphere_obj(m_scene, Material{}, Transform{}));
            }
            if (ImGui::MenuItem("Quad")) {
                add_object(Object::make_quad_obj(m_scene, Material{}, Transform{}));
            }

            if (ImGui::MenuItem("Import .obj File")) {
                auto const path = ImGui::FileDialogue(ImGui::FileDialogueMode::OPEN, "obj");
                if(!path.empty()) {
                    std::string warning;
                    auto const obj = check_error(Object::from_obj(m_scene, Material{}, path, &warning));
                    add_object(obj);

                    this->log(LogLevel::Warning, warning);
                }
            }
            if (ImGui::MenuItem("Add Light")) {
                add_light(Light { m_scene,glm::vec3(1.0f), 1.0f, Transform{}});
            }

            ImGui::EndMenu();
        }
    }
}

void EditorApplication::draw_object_panel() noexcept {
    if (ImGui::CollapsingHeader("Object Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (m_control_state.get_cur_obj()) {
            auto editable = m_control_state.get_cur_obj().value();

            // NOTE: no safety check here, cuz I'm lazy
            if (ImGui::InputText("Name", m_control_state.obj_name_buf.data(), m_control_state.obj_name_buf.size(),
                ImGuiInputTextFlags_EnterReturnsTrue)) 
            {
                editable.set_name(m_control_state.obj_name_buf.data());
            }

            auto&& gizmo_state = m_control_state.gizmo_state;
            auto trans = editable.get_transform();
            if (ImGui::TransformField("Transform", trans, gizmo_state.op, gizmo_state.mode, gizmo_state.snap, gizmo_state.snap_scale)) {
                editable.set_transform(trans);
                on_editable_change(editable, EditableChangeType::TRANSFORM);
            }

            // editable-specific fields
            if (auto const obj = editable.as<Object>()) {
                auto mat = obj->get_material();
                if (ImGui::ReflectedField("Material", mat)) {
                    obj->set_material(mat);
                    on_editable_change(editable, EditableChangeType::MATERIAL);
                }
            } else if (auto const light = editable.as<Light>()) {
                if (ImGui::ReflectedField("Light", *light)) {
                    on_editable_change(editable, EditableChangeType::OTHER);
                }
            }
        } else {
            ImGui::Text("No object selected");
        }
    }
}

void EditorApplication::draw_scene_viewport(TextureHandle render_buf) noexcept {
    if (ImGui::BeginMenuBar()) {
        ImGui::Text("Select Renderer");
        ImGui::SameLine();
        if (ImGui::Combo("##Select Renderer", &m_control_state.cur_renderer_idx,
            [](void* data, int idx, char const** out_text) {
                auto const& renderers = *static_cast<std::vector<std::unique_ptr<Renderer>> const*>(data);
                *out_text = renderers[idx]->get_name().data();
                return true;
            }, &m_renderers, m_renderers.size())) 
        {
            on_render_config_change(m_config);
        }
        
        ImGui::EndMenuBar();
    }

    if (get_cur_renderer().valid()) {
        static auto last_size = ImVec2{ 0, 0 };

        auto const v_min = ImGui::GetWindowContentRegionMin();
        auto const v_max = ImGui::GetWindowContentRegionMax();
        auto const view_size = v_max - v_min;

        if (std::abs(view_size.x - last_size.x) >= 0.01f || std::abs(view_size.y - last_size.y) >= 0.01f) {
            m_config.width = static_cast<unsigned>(view_size.x);
            m_config.height = static_cast<unsigned>(view_size.y);
            on_render_config_change(m_config);
            last_size = view_size;
        }

        check_error(render_buf->bind());
        auto uv0 = ImVec2{ 0, 0 };
        auto uv1 = ImVec2{ 1, 1 };

        if (render_buf->get_width() != m_config.width) {
            uv1.x = m_config.width / static_cast<float>(render_buf->get_width());
        }
        if (render_buf->get_height() != m_config.height) {
            uv1.y = m_config.height / static_cast<float>(render_buf->get_height());
        }
        // imgui uses top-left as origin, but OpenGL use bottom-left
        uv0.y = uv1.y;
        uv1.y = 0;

        ImGui::Image(
            render_buf->get_id(),
            view_size,
            uv0,
            uv1
        );
        render_buf->unbind();

        // draw x,y,z axis ref

        // TODO: figure out why ImGuiConfigFlags_ViewportsEnable makes these not work
        //auto const vp_size = glm::ivec2{ m_config.width, m_config.height };
        //auto const axis_origin = m_cam.viewport_to_world(glm::vec2 {30, 30}, vp_size,0.0f);
        //constexpr float axis_len = 0.01f;
        //get_debug_drawer().begin_relative(to_glm(ImGui::GetWindowPos() + v_min));
        //get_debug_drawer().draw_line_3d(m_cam, vp_size, axis_origin, axis_origin + glm::vec3{ axis_len, 0, 0 }, { 1, 0, 0 }, 2.0f, 0.0f);
        //get_debug_drawer().draw_line_3d(m_cam, vp_size, axis_origin, axis_origin + glm::vec3{ 0, axis_len, 0 }, { 0, 1, 0 }, 2.0f, 0.0f);
        //get_debug_drawer().draw_line_3d(m_cam, vp_size, axis_origin, axis_origin + glm::vec3{ 0, 0, axis_len }, { 0, 0, 1 }, 2.0f, 0.0f);
        //get_debug_drawer().end_relative();

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
                on_editable_change(m_control_state.get_cur_obj().value(), EditableChangeType::TRANSFORM);
            }
        }
    } else {
        ImGui::Text("Renderer not found");
    }
}

void EditorApplication::draw_console_panel() const noexcept {
    static EArray<LogLevel, ImVec4> const s_log_colors{
        { LogLevel::Error, ImVec4{1,0,0,1} },
        { LogLevel::Debug, ImVec4{0,1,1,0} },
        { LogLevel::Info, ImVec4{1,1,1,1} },
        { LogLevel::Warning, ImVec4{1,1,0,1} }
    };

	ImGui::BeginChild("##scroll");
    {
        for(auto&& [level, msg] : get_logs().get()) {
            ImGui::PushStyleColor(ImGuiCol_Text, s_log_colors[level]);
            ImGui::TextUnformatted(msg.data());
        	ImGui::PopStyleColor();
        }
    }
    ImGui::EndChild();
}

void EditorApplication::on_scene_opened(Scene const& scene) {
    m_cam.set_aspect(m_config.get_aspect());
    m_control_state.set_cur_obj(std::nullopt);

    for (auto&& renderer : m_renderers) {
        check_error(renderer->open_scene(scene));
    }
}

void EditorApplication::on_render_config_change(RenderConfig const& conf) {
    m_cam.set_aspect(conf.get_aspect());
    m_cam.set_fov(conf.fovy);
    for (auto&& renderer : m_renderers) {
        check_error(renderer->set_render_config(conf));
    }
}

void EditorApplication::on_mouse_leave_scene_viewport() noexcept {
    m_control_state.is_outside_view = true;
}

void EditorApplication::on_mouse_enter_scene_viewport() noexcept {
    m_control_state.is_outside_view = false;
}

void EditorApplication::on_obj_change(std::optional<EditableView> obj) noexcept {}

void EditorApplication::try_select_object() noexcept {
    auto pos = ImGui::GetMousePos();
    auto const win_pos = get_window_content_pos(k_scene_view_win_name);
    if (!win_pos) {
        this->log(LogLevel::Error, "scene view not found");
        return;
    } else if (get_cur_hovered_widget() != k_scene_view_win_name) {
        return;
    } else if (get_cur_focused_widget() != k_scene_view_win_name) {
        return;
    }

    // convert to viewport space
    pos = pos - win_pos.value();
    auto const ray = m_cam.viewport_to_ray(to_glm(pos), { m_config.width, m_config.height });
    if (auto const res = m_scene.ray_cast_editable(ray)) {
        m_control_state.set_cur_obj(*res); // note: deselection is handled by key press
    }
}

void EditorApplication::handle_input(InputEvent const& event) noexcept {
    for (auto&& action : m_input_actions) {
        action(event);
    }
}

void EditorApplication::add_object(Object obj) noexcept {
    auto const pobj = m_scene.add_object(std::move(obj));
    if(!pobj) {
        this->log(LogLevel::Error, "failed to add object");
    } else {
        on_add_editable(*pobj);
    }

    // imagine that the camera view is a sphere
    // if the scene bound is outside the sphere, move the camera to a better position
    auto const new_bbox = m_scene.get_scene_bound();
    auto const radius = glm::length(m_cam.get_eye() - m_cam.get_center());
    auto const new_radius = new_bbox.get_extent().length();
    if (new_radius > radius * 1.5f) {
        m_cam.set(m_scene.get_good_cam_start());
    }
}

void EditorApplication::add_light(Light light) noexcept {
    auto const plight = m_scene.add_light(std::move(light));
    if (!plight) {
        this->log(LogLevel::Error, "failed to add object");
    }
    else {
        on_add_editable(*plight);
    }
}

void EditorApplication::remove_editable(EditableView editable) {
    for (auto&& renderer : m_renderers) {
        check_error(renderer->on_remove_editable(editable));
    }

    if (auto const obj = editable.as<Object>()) {
        m_scene.remove_object(*obj);
    } else if (auto const light = editable.as<Light>()) {
        m_scene.remove_light(*light);
    }

    m_control_state.set_cur_obj(std::nullopt);
}

void EditorApplication::on_add_editable(EditableView editable) {
    for (auto&& renderer : m_renderers) {
        check_error(renderer->on_add_editable(editable));
    }
    
    m_control_state.set_cur_obj(editable);
}

void EditorApplication::on_editable_change(EditableView editable, EditableChangeType type) {
    for (auto&& renderer : m_renderers) {
        check_error(renderer->on_editable_change(m_control_state.get_cur_obj().value(), type));
    }
}

void EditorApplication::on_log_added() { }

auto EditorApplication::get_cur_renderer() noexcept -> Renderer& {
    if (m_control_state.cur_renderer_idx >= m_renderers.size() || m_control_state.cur_renderer_idx < 0) {
        this->log(LogLevel::Error, "the current renderer is no longer valid");
        m_control_state.cur_renderer_idx = k_default_renderer_idx;
    }

    return *m_renderers[m_control_state.cur_renderer_idx];
}

void EditorApplication::ControlState::set_cur_obj(std::optional<EditableView> obj) noexcept {
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
