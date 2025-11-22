#include "editorApplication.h"

#include <imgui_internal.h>

#include <filesystem>

#include "boundingVolume.h"
#include "camera.h"
#include "debugDrawer.h"
#include "editorRenderer.h"
#include "editorResources.h"
#include "imgui/editorFields.h"
#include "imgui/fileDialogue.h"
#include "imgui/imhelper.h"
#include "imgui/reflectedField.h"
#include "intersection.h"
#include "jsonArchive.h"
#include "light.h"
#include "renderableObject.h"
#include "scene.h"
#include "sceneObject.h"
#include "vulkanRayTracingRenderer.h"

using namespace PTS;
using namespace PTS::Editor;

static constexpr auto k_scene_setting_win_name = "Scene Settings";
static constexpr auto k_inspector_win_name = "Inspector";
static constexpr auto k_scene_view_win_name = "Scene";
static constexpr auto k_console_win_name = "Console";

EditorApplication::EditorApplication(std::string_view name, RenderConfig config)
    : GLFWApplication{name, config.width, config.height, config.min_frame_time},
      m_config{config},
      m_cam{config.fovy, config.get_aspect(), LookAtParams{}},
      m_archive{new JsonArchive} {
    // default renderers
    add_renderer(std::make_unique<EditorRenderer>(config));
    add_renderer(std::make_unique<VulkanRayTracingRenderer>(config));

    // callbacks
    get_imgui_window_info(k_scene_view_win_name).on_enter_region +=
        [this] { on_mouse_enter_scene_viewport(); };
    get_imgui_window_info(k_scene_view_win_name).on_leave_region +=
        [this] { on_mouse_leave_scene_viewport(); };

    m_scene.get_callback_list(SceneChangeType::OBJECT_ADDED) +=
        [this](Ref<SceneObject> obj) { this->on_add_oject(obj); };
    m_scene.get_callback_list(SceneChangeType::OBJECT_REMOVED) +=
        [this](Ref<SceneObject> obj) { this->on_remove_object(obj); };

    // input actions
    create_input_actions();
}

auto EditorApplication::create_input_actions() noexcept -> void {
    // initialize input actions
    auto obj_selected =
        InputActionConstraint{[this](InputEvent const&) { return m_control_state.get_cur_obj(); }};
    auto not_using_gizmo =
        InputActionConstraint{[this](InputEvent const&) { return !ImGuizmo::IsUsing(); }};
    auto scene_view_focused = InputActionConstraint{
        [this](InputEvent const&) { return get_cur_focused_widget() == k_scene_view_win_name; }};
    auto scene_view_hovered = InputActionConstraint{
        [this](InputEvent const&) { return get_cur_hovered_widget() == k_scene_view_win_name; }};
    auto initiated_in_scene_view = InputActionConstraint{[this](InputEvent const& event) {
        return event.initiated_window == k_scene_view_win_name;
    }};

    // object manipulation
    auto get_cam_accel_factor = [this]() {
        // zoom and move faster if we are far away from the origin
        auto dist = length(m_cam.get_eye());
        auto accel_factor = dist / 1.5f * get_delta_time();
        return accel_factor;
    };

    auto translate_mode = InputAction{
        {InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_W}, [this](InputEvent const& event) {
            m_control_state.gizmo_state.op = ImGuizmo::OPERATION::TRANSLATE;
        }};
    auto rotate_mode = InputAction{{InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_E},
                                   [this](InputEvent const& event) {
                                       m_control_state.gizmo_state.op = ImGuizmo::OPERATION::ROTATE;
                                   }};
    auto scale_mode = InputAction{{InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_R},
                                  [this](InputEvent const& event) {
                                      m_control_state.gizmo_state.op = ImGuizmo::OPERATION::SCALE;
                                  }};
    auto toggle_snap = InputAction{
        {InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_X}, [this](InputEvent const& event) {
            m_control_state.gizmo_state.snap = !m_control_state.gizmo_state.snap;
        }};

    auto delete_obj = InputAction{{InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_Delete},
                                  [this](InputEvent const& event) {
                                      if ((get_cur_focused_widget() == k_scene_view_win_name ||
                                           get_cur_focused_widget() == k_scene_setting_win_name) &&
                                          get_cur_focused_widget() == get_cur_hovered_widget()) {
                                          m_scene.remove_object(*m_control_state.get_cur_obj());
                                      }
                                  }};

    auto focus_on_obj = InputAction{
        {InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_F}, [this](InputEvent const& event) {
            BoundingBox local_bound;
            if (auto obj = m_control_state.get_cur_obj()->as<RenderableObject>()) {
                local_bound = obj->get_bound();
            } else if (auto light = m_control_state.get_cur_obj()->as<Light>()) {
                local_bound = BoundingBox{glm::vec3{-0.5f}, glm::vec3{0.5f}};
            }
            LookAtParams params;
            params.center =
                m_control_state.get_cur_obj()->get_transform(TransformSpace::WORLD).get_position();
            params.eye = params.center + local_bound.get_extent() * 2.0f;
            params.up = glm::vec3{0, 1, 0};
            m_cam.set(params);
        }};
    auto select_obj =
        InputAction{{InputType::MOUSE, ActionType::PRESS, ImGuiMouseButton_Left},
                    [this](InputEvent const& event) {
                        if (!m_control_state.is_outside_view &&
                            (glfwGetTime() - event.time) <= k_object_select_mouse_time) {
                            try_select_object();
                        }
                    }};
    auto deselect_obj =
        InputAction{{InputType::KEYBOARD, ActionType::PRESS, ImGuiKey_Escape},
                    [this](InputEvent const& event) { m_control_state.set_cur_obj(nullptr); }};

    auto cam_pedestal = InputAction{
        {InputType::MOUSE, ActionType::HOLD, ImGuiMouseButton_Middle},
        [this, get_cam_accel_factor](InputEvent const& event) {
            auto pedestal =
                glm::vec3{0, m_control_state.move_sensitivity * event.normalized_mouse_delta.y, 0};
            m_cam.set_delta_dolly(pedestal * std::sqrtf(get_cam_accel_factor() * 100.0f));
        }};
    auto cam_pan =
        InputAction{{InputType::MOUSE, ActionType::HOLD, ImGuiMouseButton_Left},
                    [this, get_cam_accel_factor](InputEvent const& event) {
                        auto dolly = glm::vec3{
                            -m_control_state.move_sensitivity * event.normalized_mouse_delta.x, 0,
                            m_control_state.move_sensitivity * event.normalized_mouse_delta.y};
                        m_cam.set_delta_dolly(dolly * std::sqrtf(get_cam_accel_factor() * 100.0f));
                    }};
    auto cam_rotate =
        InputAction{{InputType::MOUSE, ActionType::HOLD, ImGuiMouseButton_Right},
                    [this](InputEvent const& event) {
                        m_cam.set_delta_rotation(
                            {m_control_state.rot_sensitivity * event.normalized_mouse_delta.y,
                             m_control_state.rot_sensitivity * event.normalized_mouse_delta.x, 0});
                    }};
    auto cam_zoom =
        InputAction{{InputType::MOUSE, ActionType::SCROLL, ImGuiMouseButton_Middle},
                    [this, get_cam_accel_factor](InputEvent const& event) {
                        m_cam.set_delta_zoom(event.mouse_scroll_delta.y * get_cam_accel_factor());
                    }};

    auto on_begin_cam_pedestal =
        InputAction{{InputType::MOUSE, ActionType::PRESS, cam_pedestal.get_input().key_or_button},
                    [this](InputEvent const&) { ++m_control_state.is_changing_scene_cam; }};
    auto on_begin_cam_pan =
        InputAction{{InputType::MOUSE, ActionType::PRESS, cam_pan.get_input().key_or_button},
                    [this](InputEvent const&) { ++m_control_state.is_changing_scene_cam; }};
    auto on_begin_cam_rotate =
        InputAction{{InputType::MOUSE, ActionType::PRESS, cam_rotate.get_input().key_or_button},
                    [this](InputEvent const&) { ++m_control_state.is_changing_scene_cam; }};
    auto on_end_cam_pedestal =
        InputAction{{InputType::MOUSE, ActionType::RELEASE, cam_pedestal.get_input().key_or_button},
                    [this](InputEvent const&) {
                        m_control_state.is_changing_scene_cam =
                            std::max(m_control_state.is_changing_scene_cam - 1, 0);
                    }};
    auto on_end_cam_pan =
        InputAction{{InputType::MOUSE, ActionType::RELEASE, cam_pan.get_input().key_or_button},
                    [this](InputEvent const&) {
                        m_control_state.is_changing_scene_cam =
                            std::max(m_control_state.is_changing_scene_cam - 1, 0);
                    }};
    auto on_end_cam_rotate =
        InputAction{{InputType::MOUSE, ActionType::RELEASE, cam_rotate.get_input().key_or_button},
                    [this](InputEvent const&) {
                        m_control_state.is_changing_scene_cam =
                            std::max(m_control_state.is_changing_scene_cam - 1, 0);
                    }};

    m_input_actions = {
        translate_mode.add_constraint(obj_selected),
        rotate_mode.add_constraint(obj_selected),
        scale_mode.add_constraint(obj_selected),
        toggle_snap.add_constraint(obj_selected),
        delete_obj.add_constraint(obj_selected),
        focus_on_obj.add_constraint(scene_view_focused).add_constraint(obj_selected),
        select_obj.add_constraint(scene_view_focused).add_constraint(scene_view_hovered),
        deselect_obj.add_constraint(scene_view_focused)
            .add_constraint(scene_view_hovered)
            .add_constraint(obj_selected),
        cam_pedestal.add_constraint(initiated_in_scene_view).add_constraint(not_using_gizmo),
        cam_pan.add_constraint(initiated_in_scene_view).add_constraint(not_using_gizmo),
        cam_rotate.add_constraint(initiated_in_scene_view).add_constraint(not_using_gizmo),
        cam_zoom.add_constraint(scene_view_focused)
            .add_constraint(scene_view_hovered)
            .add_constraint(not_using_gizmo),
        on_begin_cam_pedestal.add_constraint(initiated_in_scene_view)
            .add_constraint(not_using_gizmo),
        on_begin_cam_pan.add_constraint(initiated_in_scene_view).add_constraint(not_using_gizmo),
        on_begin_cam_rotate.add_constraint(initiated_in_scene_view).add_constraint(not_using_gizmo),
        on_end_cam_pedestal.add_constraint(initiated_in_scene_view).add_constraint(not_using_gizmo),
        on_end_cam_pan.add_constraint(initiated_in_scene_view).add_constraint(not_using_gizmo),
        on_end_cam_rotate.add_constraint(initiated_in_scene_view).add_constraint(not_using_gizmo),
    };
}

auto EditorApplication::wrap_mouse_pos() noexcept -> void {
    if (m_control_state.is_changing_scene_cam) {
        auto iwx = int{}, iwy = int{};
        glfwGetWindowSize(m_window, &iwx, &iwy);
        auto const wx = static_cast<float>(iwx);
        auto const wy = static_cast<float>(iwy);
        static constexpr auto eps = 0.1f;

        auto changed{false};
        if (m_mouse_pos.x > wx + eps) {
            m_mouse_pos.x = 0;
            changed = true;
        }
        if (m_mouse_pos.y > wy + eps) {
            m_mouse_pos.y = 0;
            changed = true;
        }
        if (m_mouse_pos.x < -eps) {
            m_mouse_pos.x = wx;
            changed = true;
        }
        if (m_mouse_pos.y < -eps) {
            m_mouse_pos.y = wy;
            changed = true;
        }

        if (changed) {
            glfwSetCursorPos(m_window, m_mouse_pos.x, m_mouse_pos.y);
        }
    }
}

auto EditorApplication::add_renderer(std::unique_ptr<Renderer> renderer) noexcept -> void {
    if (!renderer) {
        this->log(LogLevel::Error, "add_renderer(): renderer is null");
        return;
    }

    // initialize renderer
    check_error(renderer->init(this));
    check_error(renderer->open_scene(m_scene));

    // editor renderer-specific initialization
    if (auto p_editor_renderer = dynamic_cast<EditorRenderer*>(renderer.get()); p_editor_renderer) {
        m_control_state.get_on_selected_obj_change_callback_list() +=
            [p_editor_renderer](auto&& obj) {
                p_editor_renderer->on_selected_editable_change(obj);
            };
    }

    m_renderers.emplace_back(std::move(renderer));
}

auto EditorApplication::on_begin_first_loop() -> void {
    GLFWApplication::on_begin_first_loop();

    // do layout initialization work
    if (ImGui::GetIO().IniFilename) {
        // if we already have a saved layout, do nothing
        if (std::filesystem::exists(ImGui::GetIO().IniFilename)) {
            return;
        }
    }

    // create an UI that covers the whole window, for docking
    auto id = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(),
                                           ImGuiDockNodeFlags_PassthruCentralNode);
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

auto EditorApplication::loop(float dt) -> void {
    // ImGuizmo
    ImGuizmo::BeginFrame();

    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
    auto const render_tex = check_error(get_cur_renderer().render(m_cam));

    // draw left panel
    if (begin_imgui_window(k_scene_setting_win_name, ImGuiWindowFlags_NoMove)) {
        draw_scene_panel();
    }
    end_imgui_window();

    // draw right panel
    if (begin_imgui_window(k_inspector_win_name, ImGuiWindowFlags_NoMove)) {
        draw_object_panel();
    }
    end_imgui_window();

    // draw bottom panel
    if (begin_imgui_window(k_console_win_name, ImGuiWindowFlags_NoMove)) {
        draw_console_panel();
    }
    end_imgui_window();

    // draw the scene view
    if (begin_imgui_window(k_scene_view_win_name, ImGuiWindowFlags_NoScrollWithMouse |
                                                      ImGuiWindowFlags_NoMove |
                                                      ImGuiWindowFlags_MenuBar)) {
        draw_scene_viewport(render_tex);
    }
    end_imgui_window();

    check_error(get_cur_renderer().draw_imgui());

    wrap_mouse_pos();
}

auto EditorApplication::quit(int code) -> void {
    std::exit(code);
}

auto EditorApplication::draw_scene_panel() noexcept -> void {
    ImGui::TextUnformatted(k_editor_tutorial_text);
    ImGui::Separator();

    if (ImGui::Button("Open Scene")) {
        auto const path = FileDialogue(ImGui::FileDialogueMode::OPEN, m_archive->get_ext().data());
        if (!path.empty()) {
            check_error(m_archive->load_file(path, m_scene, m_cam));
            on_scene_opened(m_scene);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Save Scene")) {
        auto const path = FileDialogue(ImGui::FileDialogueMode::SAVE, m_archive->get_ext().data());
        if (!path.empty()) {
            check_error(m_archive->save_file(m_scene, m_cam, path));
        }
    }

    if (ImGui::CollapsingHeader("Editor Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Move Sensitivity");
        ImGui::SliderFloat("##Move Sensitivity", &m_control_state.move_sensitivity, 1.0f, 10.0f);
        ImGui::Text("Rotate Sensitivity");
        ImGui::SliderFloat("##Rotate Sensitivity", &m_control_state.rot_sensitivity, 2.0f, 100.0f);

        auto const prev_disable_log_flush = m_control_state.disable_log_flush;
        ImGui::Checkbox("Disable Log Flush", &m_control_state.disable_log_flush);
        if (m_control_state.disable_log_flush) {
            m_log_flush_interval = std::numeric_limits<float>::infinity();
        } else {
            if (prev_disable_log_flush) {
                m_log_flush_timer = 0.0f;
            }
        }

        ImGui::BeginDisabled(m_control_state.disable_log_flush);
        {
            ImGui::Text("Log Flush Interval");
            ImGui::SameLine();

            ImGui::SliderFloat("##Log Flush Interval", &m_log_flush_interval, 1.0f, 60.0f);
        }
        ImGui::EndDisabled();
    }

    if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Unlimited FPS", &m_control_state.unlimited_fps);
        if (m_control_state.unlimited_fps) {
            set_min_frame_time(0.0f);
        }

        ImGui::BeginDisabled(m_control_state.unlimited_fps);
        {
            auto cap = m_control_state.unlimited_fps ? std::numeric_limits<float>::infinity()
                                                     : 1 / get_min_frame_time();

            ImGui::Text("FPS Cap");
            ImGui::SameLine();
            if (ImGui::SliderFloat("##FPS Cap", &cap, 1.0f, 1000.0f)) {
                set_min_frame_time(1 / cap);
            }
        }
        ImGui::EndDisabled();

        ImGui::Text("Current FPS: %.2f", 1.0f / get_delta_time());
    }

    // draw scene object list
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Scene Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::BeginListBox("##Scene Objects", {0, 200})) {
            for (auto editable_ref : m_scene.get_editables()) {
                auto&& editable = editable_ref.get();
                if (!editable.is_editable()) {
                    // should not be possible, objects with _NoEdit flag should not be in the
                    // editable list
                    this->log(LogLevel::Error,
                              "Editable with _NoEdit flag found in m_scene.get_editables()");
                    continue;
                }

                auto display_name = std::string{editable.get_name()};
                if (!(editable.get_edit_flags() & Visible)) {
                    display_name += " (hidden)";
                    ImGui::PushStyleColor(ImGuiCol_Text, {0.5f, 0.5f, 0.5f, 1.0f});
                }
                if (!(editable.get_edit_flags() & Selectable)) {
                    display_name += " (unselectable)";
                    ImGui::PushStyleColor(ImGuiCol_Text, {0.5f, 0.5f, 0.5f, 1.0f});
                }

                if (ImGui::Selectable(display_name.c_str(),
                                      m_control_state.get_cur_obj() == &editable)) {
                    m_control_state.set_cur_obj(&editable);
                }

                if (!(editable.get_edit_flags() & Visible)) {
                    ImGui::PopStyleColor();
                }
                if (!(editable.get_edit_flags() & Selectable)) {
                    ImGui::PopStyleColor();
                }
            }

            ImGui::EndListBox();
        }
        ImGui::Spacing();

        if (ImGui::BeginMenu("Add Object")) {
            if (ImGui::MenuItem("Triangle")) {
                m_scene.emplace_object<RenderableObject>(RenderableObject::make_triangle_obj(
                    m_scene, k_editable_flags, Material{}, Transform{}));
            }
            if (ImGui::MenuItem("Cube")) {
                m_scene.emplace_object<RenderableObject>(RenderableObject::make_cube_obj(
                    m_scene, k_editable_flags, Material{}, Transform{}));
            }
            if (ImGui::MenuItem("Sphere")) {
                m_scene.emplace_object<RenderableObject>(RenderableObject::make_sphere_obj(
                    m_scene, k_editable_flags, Material{}, Transform{}));
            }
            if (ImGui::MenuItem("Quad")) {
                m_scene.emplace_object<RenderableObject>(RenderableObject::make_quad_obj(
                    m_scene, k_editable_flags, Material{}, Transform{}));
            }

            if (ImGui::MenuItem("Import .obj File")) {
                auto const path = FileDialogue(ImGui::FileDialogueMode::OPEN, "obj");
                if (!path.empty()) {
                    std::string warning;
                    auto obj = check_error(RenderableObject::from_obj(m_scene, k_editable_flags,
                                                                      Material{}, path, &warning));
                    m_scene.emplace_object<RenderableObject>(std::move(obj));
                    this->log(LogLevel::Warning, warning);
                }
            }
            if (ImGui::MenuItem("Add Light")) {
                m_scene.emplace_object<Light>(m_scene, Transform{}, k_editable_flags,
                                              LightType::Point, glm::vec3(1.0f), 1.0f);
            }

            ImGui::EndMenu();
        }
    }
}

auto EditorApplication::draw_object_panel() noexcept -> void {
    if (ImGui::CollapsingHeader("Object Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (auto const editable = m_control_state.get_cur_obj()) {
            // NOTE: no safety check here, cuz I'm lazy
            if (ImGui::InputText("Name", m_control_state.obj_name_buf.data(),
                                 m_control_state.obj_name_buf.size(),
                                 ImGuiInputTextFlags_EnterReturnsTrue)) {
                editable->set_name(m_control_state.obj_name_buf.data());
            }

            auto&& gizmo_state = m_control_state.gizmo_state;
            auto trans = editable->get_transform(TransformSpace::LOCAL);
            if (ImGui::TransformField("Transform", trans, gizmo_state.op, gizmo_state.mode,
                                      gizmo_state.snap, gizmo_state.snap_scale)) {
                editable->set_transform(trans, TransformSpace::LOCAL);
            }

            // editable-specific fields
            if (auto const obj = editable->as<RenderableObject>()) {
                ImGui::ReflectedField("Object Data", *obj);
            } else if (auto const light = editable->as<Light>()) {
                ImGui::ReflectedField("Light", *light);
            }
        } else {
            ImGui::Text("No object selected");
        }
    }
}

auto EditorApplication::draw_scene_viewport(TextureHandle render_buf) noexcept -> void {
    if (ImGui::BeginMenuBar()) {
        ImGui::Text("Select Renderer");
        ImGui::SameLine();
        if (ImGui::Combo(
                "##Select Renderer", &m_control_state.cur_renderer_idx,
                [](void* data, int idx, char const** out_text) {
                    auto const& renderers =
                        *static_cast<std::vector<std::unique_ptr<Renderer>> const*>(data);
                    *out_text = renderers[idx]->get_name().data();
                    return true;
                },
                &m_renderers, m_renderers.size())) {
            on_render_config_change(m_config);
        }

        ImGui::EndMenuBar();
    }

    if (get_cur_renderer().valid()) {
        static auto last_size = ImVec2{0, 0};

        auto const v_min = ImGui::GetWindowContentRegionMin();
        auto const v_max = ImGui::GetWindowContentRegionMax();
        auto const view_size = v_max - v_min;

        if (std::abs(view_size.x - last_size.x) >= 0.01f ||
            std::abs(view_size.y - last_size.y) >= 0.01f) {
            m_config.width = static_cast<unsigned>(view_size.x);
            m_config.height = static_cast<unsigned>(view_size.y);
            on_render_config_change(m_config);
            last_size = view_size;
        }

        check_error(render_buf->bind());
        auto uv0 = ImVec2{0, 0};
        auto uv1 = ImVec2{1, 1};

        if (render_buf->get_width() != m_config.width) {
            uv1.x = m_config.width / static_cast<float>(render_buf->get_width());
        }
        if (render_buf->get_height() != m_config.height) {
            uv1.y = m_config.height / static_cast<float>(render_buf->get_height());
        }
        // imgui uses top-left as origin, but OpenGL use bottom-left
        uv0.y = uv1.y;
        uv1.y = 0;

        ImGui::Image(render_buf->get_id(), view_size, uv0, uv1);
        render_buf->unbind();

        // draw x,y,z axis ref

        // TODO: figure out why ImGuiConfigFlags_ViewportsEnable makes these not work
        // auto const vp_size = glm::ivec2{ m_config.width, m_config.height };
        // auto const axis_origin = m_cam.viewport_to_world(glm::vec2 {30, 30}, vp_size,0.0f);
        // constexpr float axis_len = 0.01f;
        // get_debug_drawer().begin_relative(to_glm(ImGui::GetWindowPos() + v_min));
        // get_debug_drawer().draw_line_3d(m_cam, vp_size, axis_origin, axis_origin + glm::vec3{
        // axis_len, 0, 0 }, { 1, 0, 0 }, 2.0f, 0.0f); get_debug_drawer().draw_line_3d(m_cam,
        // vp_size, axis_origin, axis_origin + glm::vec3{ 0, axis_len, 0 }, { 0, 1, 0 }, 2.0f,
        // 0.0f); get_debug_drawer().draw_line_3d(m_cam, vp_size, axis_origin, axis_origin +
        // glm::vec3{ 0, 0, axis_len }, { 0, 0, 1 }, 2.0f, 0.0f); get_debug_drawer().end_relative();

        // draw gizmos
        if (m_control_state.get_cur_obj()) {
            auto const win = ImGui::GetCurrentWindow();
            auto const& gizmo_state = m_control_state.gizmo_state;

            ImGuizmo::SetDrawlist(win->DrawList);
            ImGuizmo::SetRect(win->Pos.x, win->Pos.y, win->Size.x, win->Size.y);

            auto mat =
                m_control_state.get_cur_obj()->get_transform(TransformSpace::LOCAL).get_matrix();
            if (Manipulate(value_ptr(m_cam.get_view()), value_ptr(m_cam.get_projection()),
                           gizmo_state.op, gizmo_state.mode, value_ptr(mat), nullptr,
                           gizmo_state.snap ? value_ptr(gizmo_state.snap_scale) : nullptr)) {
                m_control_state.get_cur_obj()->set_transform(Transform{mat}, TransformSpace::LOCAL);
            }
        }
    } else {
        ImGui::Text("Renderer not found");
    }
}

auto EditorApplication::draw_console_panel() const noexcept -> void {
    static EArray<LogLevel, ImVec4> const s_log_colors{{LogLevel::Error, ImVec4{1, 0, 0, 1}},
                                                       {LogLevel::Debug, ImVec4{0, 1, 1, 0}},
                                                       {LogLevel::Info, ImVec4{1, 1, 1, 1}},
                                                       {LogLevel::Warning, ImVec4{1, 1, 0, 1}}};

    ImGui::BeginChild("##scroll");
    {
        for (auto&& [level, msg] : get_logs().get()) {
            ImGui::PushStyleColor(ImGuiCol_Text, s_log_colors[level]);
            ImGui::TextUnformatted(msg.data());
            ImGui::PopStyleColor();
        }
    }
    ImGui::EndChild();
}

auto EditorApplication::on_scene_opened(Scene& scene) -> void {
    m_cam.set_aspect(m_config.get_aspect());
    m_control_state.set_cur_obj(nullptr);

    for (auto&& renderer : m_renderers) {
        check_error(renderer->open_scene(scene));
    }
}

auto EditorApplication::on_render_config_change(RenderConfig const& conf) -> void {
    m_cam.set_aspect(conf.get_aspect());
    m_cam.set_fov(conf.fovy);
    for (auto&& renderer : m_renderers) {
        check_error(renderer->set_render_config(conf));
    }
}

auto EditorApplication::on_mouse_leave_scene_viewport() noexcept -> void {
    m_control_state.is_outside_view = true;
}

auto EditorApplication::on_mouse_enter_scene_viewport() noexcept -> void {
    m_control_state.is_outside_view = false;
}

auto EditorApplication::try_select_object() noexcept -> void {
    auto pos = ImGui::GetMousePos();
    auto const win_pos = get_window_content_pos(k_scene_view_win_name);
    if (!win_pos) {
        this->log(LogLevel::Error, "scene view not found");
        return;
    }
    if (get_cur_hovered_widget() != k_scene_view_win_name) {
        return;
    }
    if (get_cur_focused_widget() != k_scene_view_win_name) {
        return;
    }

    // convert to viewport space
    pos = pos - win_pos.value();
    auto const ray = m_cam.viewport_to_ray(to_glm(pos), {m_config.width, m_config.height});
    if (auto const res = m_scene.ray_cast_editable(ray)) {
        m_control_state.set_cur_obj(res);  // note: deselection is handled by key press
    }
}

auto EditorApplication::handle_input(InputEvent const& event) noexcept -> void {
    for (auto&& action : m_input_actions) {
        action(event);
    }
}

auto EditorApplication::on_remove_object(Ref<SceneObject> obj) -> void {
    if (obj.get().is_editable()) {
        m_control_state.set_cur_obj(nullptr);
    }
}

auto EditorApplication::on_add_oject(Ref<SceneObject> obj) -> void {
    if (obj.get().is_editable()) {
        m_control_state.set_cur_obj(&obj.get());
        // adjust camera if needed
        // imagine that the camera view is a sphere
        // if the scene bound is outside the sphere, move the camera to a better position
        auto const new_bbox = m_scene.get_scene_bound();
        auto const radius = length(m_cam.get_eye() - m_cam.get_center());
        auto const new_radius = length(new_bbox.get_extent());
        if (new_radius > radius * 1.5f) {
            m_cam.set(m_scene.get_good_cam_start());
        }
    }
}

auto EditorApplication::on_log_added() -> void {
}

auto EditorApplication::get_cur_renderer() noexcept -> Renderer& {
    if (m_control_state.cur_renderer_idx >= m_renderers.size() ||
        m_control_state.cur_renderer_idx < 0) {
        this->log(LogLevel::Error, "the current renderer is no longer valid");
        m_control_state.cur_renderer_idx = k_default_renderer_idx;
    }

    return *m_renderers[m_control_state.cur_renderer_idx];
}

auto EditorApplication::ControlState::set_cur_obj(ObserverPtr<SceneObject> obj) noexcept -> void {
    if (obj == m_cur_obj) {
        return;
    }

    if (ImGuizmo::IsUsing()) {
        // let gizmos finish
        return;
    }

    m_cur_obj = obj;
    m_on_selected_obj_change_callback_list(obj);
    if (obj) {
        std::fill(obj_name_buf.begin(), obj_name_buf.end(), '\0');
        std::copy(obj->get_name().begin(), obj->get_name().end(), obj_name_buf.begin());
    }
}
