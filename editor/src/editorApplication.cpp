#include "editorApplication.h"

#include <core/imgui/fileDialogue.h>
#include <core/imgui/imhelper.h>
#include <core/imgui/reflectedField.h>
#include <core/legacy/boundingVolume.h>
#include <core/legacy/camera.h>
#include <core/legacy/intersection.h>
#include <core/legacy/jsonArchive.h>
#include <core/legacy/light.h>
#include <core/legacy/renderableObject.h>
#include <core/legacy/scene.h>
#include <core/legacy/sceneObject.h>
#include <core/logging.h>
#include <imgui_internal.h>
#include <spdlog/sinks/ringbuffer_sink.h>

#include <cstring>
#include <filesystem>
#include <glm/gtc/type_ptr.hpp>

#include "debugDrawer.h"
#include "editorResources.h"
#include "imgui/editorFields.h"

using namespace PTS;
using namespace PTS::Editor;

static constexpr auto k_scene_setting_win_name = "Scene Settings";
static constexpr auto k_inspector_win_name = "Inspector";
static constexpr auto k_scene_view_win_name = "Scene";
static constexpr auto k_console_win_name = "Console";
static constexpr auto k_console_log_buffer_size = 1024;

EditorApplication::EditorApplication(std::string_view name, RenderConfig config,
                                     pts::LoggingManager& logging_manager,
                                     pts::PluginManager& plugin_manager)
    : GLFWApplication{name, config.width, config.height, config.min_frame_time},
      m_config{config},
      m_cam{config.fovy, config.get_aspect(), LookAtParams{}},
      m_archive{new JsonArchive},
      m_logging_manager{&logging_manager},
      m_plugin_manager{&plugin_manager} {
    m_vk_context = std::make_unique<VulkanContext>(get_vk_instance(), get_vk_surface());
    init_imgui_vulkan(m_vk_context->physical_device(), m_vk_context->device(),
                      m_vk_context->queue_family(), m_vk_context->queue());

    m_render_graph =
        std::make_unique<RenderGraphHost>(m_vk_context->physical_device(), m_vk_context->device(),
                                          m_vk_context->queue(), m_vk_context->queue_family());
    m_render_graph->resize(m_config.width, m_config.height);
    m_renderer_host_api.render_graph_api = m_render_graph->api();
    m_renderer_host_api.render_world_api = nullptr;

    m_renderer_plugin = m_plugin_manager->get_plugin_instance("editor.renderer");
    if (m_renderer_plugin) {
        m_renderer_interface = static_cast<RendererPluginInterfaceV1*>(
            m_plugin_manager->query_interface(m_renderer_plugin, RENDERER_PLUGIN_INTERFACE_V1_ID));
    }
    if (!m_renderer_interface) {
        log(pts::LogLevel::Error, "Renderer plugin interface not found");
    }

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

    // logging
    m_console_log_sink =
        std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(k_console_log_buffer_size);
    m_logging_manager->add_sink(m_console_log_sink);

    log(pts::LogLevel::Info, "EditorApplication created");
}

EditorApplication::~EditorApplication() {
    m_render_graph.reset();
    // Ensure Vulkan/ImGui teardown while renderer device is still alive.
    shutdown_imgui_vulkan();
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
    if (m_renderer_interface && m_renderer_interface->build_graph && m_render_graph) {
        PtsFrameParams frame{};
        frame.frame_index = m_frame_index++;
        frame.time_seconds = get_time();
        frame.wall_time = get_time();

        PtsViewParams view{};
        std::memcpy(view.view, glm::value_ptr(m_cam.get_view()), sizeof(view.view));
        std::memcpy(view.proj, glm::value_ptr(m_cam.get_projection()), sizeof(view.proj));
        auto view_proj = m_cam.get_projection() * m_cam.get_view();
        std::memcpy(view.view_proj, glm::value_ptr(view_proj), sizeof(view.view_proj));
        std::memset(view.prev_view_proj, 0, sizeof(view.prev_view_proj));
        auto cam_pos = m_cam.get_eye();
        view.camera_pos[0] = cam_pos.x;
        view.camera_pos[1] = cam_pos.y;
        view.camera_pos[2] = cam_pos.z;
        view.jitter_xy[0] = 0.0f;
        view.jitter_xy[1] = 0.0f;
        view.dt_seconds = dt;
        view.frame_index = static_cast<uint32_t>(frame.frame_index);
        view.viewport_w = m_config.width;
        view.viewport_h = m_config.height;
        view.near_plane = 0.1f;
        view.far_plane = 100000.0f;

        PtsFrameIO io{};
        io.output = m_render_graph->output_texture();

        m_render_graph->set_current();
        m_renderer_interface->build_graph(&m_renderer_host_api, &frame, &view, &io);
        m_render_graph->clear_current();
    }

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
        draw_scene_viewport();
    }
    end_imgui_window();

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
                    this->log(pts::LogLevel::Error,
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
                    this->log(pts::LogLevel::Warning, warning);
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

auto EditorApplication::draw_scene_viewport() noexcept -> void {
    if (ImGui::BeginMenuBar()) {
        ImGui::TextUnformatted("Renderer: editor.renderer");
        ImGui::EndMenuBar();
    }

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

    if (m_render_graph && m_render_graph->output_imgui_id()) {
        ImGui::Image(m_render_graph->output_imgui_id(), view_size);
    } else {
        ImGui::TextUnformatted("Renderer output not available");
    }

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

        auto mat = m_control_state.get_cur_obj()->get_transform(TransformSpace::LOCAL).get_matrix();
        if (Manipulate(value_ptr(m_cam.get_view()), value_ptr(m_cam.get_projection()),
                       gizmo_state.op, gizmo_state.mode, value_ptr(mat), nullptr,
                       gizmo_state.snap ? value_ptr(gizmo_state.snap_scale) : nullptr)) {
            m_control_state.get_cur_obj()->set_transform(Transform{mat}, TransformSpace::LOCAL);
        }
    }
}

auto EditorApplication::draw_console_panel() const noexcept -> void {
    auto color = [](spdlog::level::level_enum lvl) -> ImVec4 {
        switch (lvl) {
            case spdlog::level::err:
                return {1, 0, 0, 1};
            case spdlog::level::warn:
                return {1, 1, 0, 1};
            case spdlog::level::info:
                return {1, 1, 1, 1};
            case spdlog::level::debug:
                return {0, 1, 1, 1};
            case spdlog::level::trace:
                return {0.7f, 0.7f, 0.7f, 1};
            case spdlog::level::critical:
                return {1, 0, 1, 1};
            default:
                return {1, 1, 1, 1};
        }
    };

    ImGui::BeginChild("##scroll");
    {
        auto msgs = m_console_log_sink->last_raw();
        for (auto&& m : msgs) {
            ImGui::PushStyleColor(ImGuiCol_Text, color(m.level));
            ImGui::TextUnformatted(m.payload.data(), m.payload.data() + m.payload.size());
            ImGui::PopStyleColor();
        }
    }
    ImGui::EndChild();
}

auto EditorApplication::on_scene_opened(Scene& scene) -> void {
    m_cam.set_aspect(m_config.get_aspect());
    m_control_state.set_cur_obj(nullptr);
}

auto EditorApplication::on_render_config_change(RenderConfig const& conf) -> void {
    m_cam.set_aspect(conf.get_aspect());
    m_cam.set_fov(conf.fovy);
    if (m_render_graph) {
        m_render_graph->resize(conf.width, conf.height);
    }
    if (m_renderer_interface && m_renderer_interface->on_resize) {
        m_renderer_interface->on_resize(conf.width, conf.height);
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
        this->log(pts::LogLevel::Error, "scene view not found");
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
