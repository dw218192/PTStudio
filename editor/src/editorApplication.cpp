#include "editorApplication.h"

#include <core/commandLine.h>
#include <core/components/imguiComponent.h>
#include <core/components/inputComponent.h>
#include <core/diagnostics.h>
#include <core/imgui/fileDialogue.h>
#include <core/rendering/passContext.h>
#include <core/rendering/rendererConfig.h>
#include <core/rendering/sceneLoader.h>
#include <core/rendering/scenePass.h>
#include <core/rendering/webgpuContext.h>
#include <core/rendering/windowing.h>
#include <imgui_internal.h>
// clang-format off
#include <ImGuizmo.h>  // must follow imgui.h
// clang-format on
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <spdlog/sinks/ringbuffer_sink.h>

#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdexcept>

#include "editorResources.h"
#include "passes/forward_pass.h"
#include "passes/grid_pass.h"

using namespace pts;
using namespace pts::editor;

static constexpr auto k_scene_setting_win_name = "Scene Settings";
static constexpr auto k_inspector_win_name = "Inspector";
static constexpr auto k_scene_view_win_name = "Scene";
static constexpr auto k_console_win_name = "Console";
static constexpr auto k_console_log_buffer_size = 1024;

static const std::vector<rendering::RendererConfig> kRendererConfigs = {
    {"Forward",
     {
         [] { return std::make_unique<ForwardPass>(); },
         [] { return std::make_unique<GridPass>(); },
     }},
};

EditorApplication::EditorApplication(std::string_view name, pts::LoggingManager& logging_manager)
    : WindowedApplication{name, logging_manager} {
    create_input_actions();

    m_console_log_sink =
        std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(k_console_log_buffer_size);
    get_logging_manager().add_sink(m_console_log_sink);

    log(pts::LogLevel::Info, "EditorApplication created");
}

EditorApplication::~EditorApplication() {
    revoke_stage_listener();
    m_passes.clear();
    m_world.clear();
    m_stage.Reset();
    m_input.reset();
    m_imgui.reset();
}

void EditorApplication::StageListener::handle(const pxr::UsdNotice::ObjectsChanged& notice,
                                              const pxr::UsdStageWeakPtr& /*sender*/) {
    PRECONDITION(cb);
    cb(ctx, notice);
}

void EditorApplication::register_stage_listener() {
    revoke_stage_listener();
    if (!m_stage) return;

    m_stage_listener.ctx = this;
    m_stage_listener.cb = [](void* self, const pxr::UsdNotice::ObjectsChanged& notice) {
        static_cast<EditorApplication*>(self)->on_objects_changed(notice);
    };

    m_listener_key = pxr::TfNotice::Register(pxr::TfCreateWeakPtr(&m_stage_listener),
                                             &StageListener::handle, m_stage);
}

void EditorApplication::revoke_stage_listener() {
    if (m_listener_key.IsValid()) {
        pxr::TfNotice::Revoke(m_listener_key);
    }
    m_stage_listener.ctx = nullptr;
    m_stage_listener.cb = nullptr;
    m_dirty_xform_paths.clear();
    m_needs_full_resync = false;
}

void EditorApplication::on_objects_changed(const pxr::UsdNotice::ObjectsChanged& notice) {
    // Resynced paths require full re-extraction
    for (const auto& path : notice.GetResyncedPaths()) {
        if (path == pxr::SdfPath::AbsoluteRootPath()) continue;
        m_needs_full_resync = true;
        return;
    }

    // Info-only changes: check if they are xform properties
    for (const auto& path : notice.GetChangedInfoOnlyPaths()) {
        // Property paths have a prim parent; check if the property is xform-related
        if (!path.IsPrimPath() && !path.IsPropertyPath()) continue;
        auto prim_path = path.IsPropertyPath() ? path.GetPrimPath() : path;
        m_dirty_xform_paths.push_back(prim_path.GetString());
    }
}

void EditorApplication::process_dirty_prims() {
    if (!m_stage) return;

    auto const& device = webgpu_context()->device();

    if (m_needs_full_resync) {
        std::string selected_prim_path;
        if (m_selected_object >= 0 &&
            m_selected_object < static_cast<int>(m_world.objects.size())) {
            selected_prim_path = m_world.objects[m_selected_object].prim_path;
        }

        m_world.clear();
        m_selected_object = -1;
        rendering::populate_from_stage(m_world, m_stage, device);

        if (!selected_prim_path.empty()) {
            for (int i = 0; i < static_cast<int>(m_world.objects.size()); ++i) {
                if (m_world.objects[i].prim_path == selected_prim_path) {
                    m_selected_object = i;
                    break;
                }
            }
        }

        log(LogLevel::Info, "Full resync: {} objects", m_world.objects.size());
        m_needs_full_resync = false;
        m_dirty_xform_paths.clear();
        return;
    }

    if (m_dirty_xform_paths.empty()) return;

    // Update transforms for dirty prims
    for (const auto& dirty_path : m_dirty_xform_paths) {
        for (auto& obj : m_world.objects) {
            if (obj.prim_path != dirty_path) continue;

            auto prim = m_stage->GetPrimAtPath(pxr::SdfPath(obj.prim_path));
            INVARIANT_MSG(prim.IsValid(),
                          "prim_path on RenderObject must reference a valid USD prim");

            pxr::UsdGeomXformable xformable(prim);
            if (!xformable) continue;

            pxr::GfMatrix4d xf =
                xformable.ComputeLocalToWorldTransform(pxr::UsdTimeCode::Default());
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c) obj.transform[c][r] = static_cast<float>(xf[r][c]);
            break;
        }
    }
    m_dirty_xform_paths.clear();
}

void EditorApplication::normalize_xform_ops(const std::string& prim_path) {
    PRECONDITION(m_stage);
    auto prim = m_stage->GetPrimAtPath(pxr::SdfPath(prim_path));
    INVARIANT_MSG(prim.IsValid(), "prim_path on RenderObject must reference a valid USD prim");

    pxr::UsdGeomXformable xformable(prim);
    if (!xformable) return;

    bool reset_xform_stack = false;
    auto ops = xformable.GetOrderedXformOps(&reset_xform_stack);
    if (ops.size() == 1 && ops[0].GetOpType() == pxr::UsdGeomXformOp::TypeTransform) return;

    auto xf = xformable.ComputeLocalToWorldTransform(pxr::UsdTimeCode::Default());
    xformable.ClearXformOpOrder();
    xformable.AddTransformOp().Set(xf);
}

void EditorApplication::register_args(CommandLine& cli) {
    WindowedApplication::register_args(cli);
    cli.add_flag("quit-on-start", "Quit the application after starting, useful for testing");
}

void EditorApplication::process_args(const CommandLine& cli) {
    WindowedApplication::process_args(cli);
    m_app_config.quit_on_start = cli.get_flag("quit-on-start");
    if (m_app_config.quit_on_start && viewport()) {
        viewport()->request_close();
    }
}

void EditorApplication::on_ready() {
    auto const& device = webgpu_context()->device();

    // ImGui + Input
    m_imgui =
        std::make_unique<ImGuiComponent>(*viewport(), *webgpu_context(), get_logging_manager());
    m_input = std::make_unique<InputComponent>(*viewport());
    m_input->set_handler([this](const InputEvent& e) { handle_input(e); });

    m_imgui->get_window_info(k_scene_view_win_name).on_enter_region.connect([this] {
        on_mouse_enter_scene_viewport();
    });
    m_imgui->get_window_info(k_scene_view_win_name).on_leave_region.connect([this] {
        on_mouse_leave_scene_viewport();
    });

    // ── Rendering init ──

    // Frame graph
    m_frame_graph = std::make_unique<rendering::FrameGraph>(
        device, get_logging_manager().get_logger_shared("frame_graph"));

    // Load default scene
    auto usda = editor_resources::get_resource("scenes/default.usda");
    if (usda) {
        auto layer = pxr::SdfLayer::CreateAnonymous(".usda");
        layer->ImportFromString(std::string{*usda});
        m_stage = pxr::UsdStage::Open(layer);
        rendering::populate_from_stage(m_world, m_stage, device);
        register_stage_listener();
        log(LogLevel::Info, "Loaded default scene ({} objects)", m_world.objects.size());
    } else {
        log(LogLevel::Warning, "Missing embedded resource: scenes/default.usda");
    }

    // Set up renderer passes
    set_renderer_config(0);

    // Camera defaults
    m_camera.set_target({0.0f, 0.0f, 0.0f});
    m_camera.set_distance(3.0f);
    m_camera.set_fov_y(60.0f);

    if (m_app_config.quit_on_start) {
        viewport()->request_close();
    }
}

void EditorApplication::set_renderer_config(size_t index) {
    PRECONDITION(index < kRendererConfigs.size());
    m_passes.clear();
    m_passes.reserve(kRendererConfigs[index].pass_factories.size());
    for (auto& factory : kRendererConfigs[index].pass_factories) {
        m_passes.push_back(factory());
    }
    auto& device = webgpu_context()->device();
    for (auto& pass : m_passes) {
        pass->setup(device);
    }
    m_active_config_index = index;
}

void EditorApplication::update(float /*dt*/) {
    // Input polling and ImGui drawing happen in render() to ensure proper
    // synchronization with ImGui::NewFrame() and the FrameGraph.
}

void EditorApplication::render(FrameContext& ctx) {
    if (!m_imgui) return;
    if (viewport() && viewport()->should_close()) return;

    // Process deferred USD change notifications before rendering
    process_dirty_prims();

    auto scope = m_imgui->frame_scope();
    ImGuizmo::BeginFrame();

    // Poll input — prev_hovered_widget makes this order-independent from UI drawing
    m_input->poll(get_time(), window_width(), window_height(), m_imgui->prev_hovered_widget());

    if (m_first_frame) {
        setup_docking_layout();
        m_first_frame = false;
    }

    // Draw UI
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                 ImGuiDockNodeFlags_PassthruCentralNode);

    if (m_imgui->begin_window(k_scene_setting_win_name, ImGuiWindowFlags_NoMove)) {
        draw_scene_panel();
    }
    m_imgui->end_window();

    if (m_imgui->begin_window(k_inspector_win_name, ImGuiWindowFlags_NoMove)) {
        draw_object_panel();
    }
    m_imgui->end_window();

    if (m_imgui->begin_window(k_console_win_name, ImGuiWindowFlags_NoMove)) {
        draw_console_panel();
    }
    m_imgui->end_window();

    if (m_imgui->begin_window(k_scene_view_win_name, ImGuiWindowFlags_NoScrollWithMouse |
                                                         ImGuiWindowFlags_NoMove |
                                                         ImGuiWindowFlags_MenuBar)) {
        draw_scene_viewport();
    }
    m_imgui->end_window();

    // ── Frame graph ──
    auto const& device = ctx.device();
    auto queue = device.queue();

    m_frame_graph->begin_frame();

    bool has_viewport = m_viewport_width > 0 && m_viewport_height > 0;

    rendering::ResourceHandle scene_color_handle;

    if (has_viewport) {
        float aspect = static_cast<float>(m_viewport_width) / static_cast<float>(m_viewport_height);
        auto view_mat = m_camera.view_matrix();
        auto proj_mat = m_camera.projection_matrix(aspect);

        rendering::PassContext pass_ctx{
            device,   queue,    m_camera,   m_world, m_viewport_width, m_viewport_height,
            view_mat, proj_mat, get_time(), 0,
        };

        for (auto& pass : m_passes) {
            if (pass->is_ready()) {
                pass->add_to_frame_graph(*m_frame_graph, pass_ctx);
            }
        }

        // Look up scene_color resource that passes created
        rendering::TextureDesc color_desc;
        color_desc.width = m_viewport_width;
        color_desc.height = m_viewport_height;
        color_desc.format = WGPUTextureFormat_RGBA8Unorm;
        color_desc.clear_color = {0.15, 0.15, 0.18, 1.0};
        scene_color_handle = m_frame_graph->find_or_create("scene_color", color_desc);
    }

    // ImGui overlay pass
    auto imgui_builder = m_frame_graph->add_pass("imgui")
                             .color(ctx.surface_view(), WGPUColor{0.08, 0.08, 0.12, 1.0})
                             .present();
    if (has_viewport && scene_color_handle.is_valid()) {
        imgui_builder.read(scene_color_handle);
    }
    imgui_builder.execute([&](WGPURenderPassEncoder pass) { scope.render_into(pass); });

    m_frame_graph->compile();
    m_frame_graph->execute(ctx.encoder());

    // ── GPU picking readback ──
    m_picking_readback.tick();

    if (auto picked_id = m_picking_readback.try_read_u32()) {
        if (*picked_id == UINT32_MAX) {
            m_selected_object = -1;
        } else if (*picked_id < static_cast<uint32_t>(m_world.objects.size())) {
            m_selected_object = static_cast<int>(*picked_id);
            if (m_stage) {
                normalize_xform_ops(m_world.objects[m_selected_object].prim_path);
            }
        }
    }

    if (m_pick_requested && has_viewport && !m_picking_readback.is_pending()) {
        rendering::TextureDesc picking_desc;
        picking_desc.width = m_viewport_width;
        picking_desc.height = m_viewport_height;
        picking_desc.format = WGPUTextureFormat_R32Uint;
        picking_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                                           WGPUTextureUsage_CopySrc);
        picking_desc.clear_color = {static_cast<double>(UINT32_MAX), 0, 0, 0};
        auto picking_handle = m_frame_graph->find_or_create("picking_ids", picking_desc);
        auto picking_ref = m_frame_graph->get_texture_ref(picking_handle);

        if (picking_ref && m_pick_x < m_viewport_width && m_pick_y < m_viewport_height) {
            m_picking_readback.request(ctx.encoder(), picking_ref.texture(), m_pick_x, m_pick_y,
                                       device.handle(), device.instance());
            m_pick_requested = false;
        } else {
            m_pick_requested = false;
        }
    }

    // Store scene color ref for next frame's ImGui::Image
    if (has_viewport && scene_color_handle.is_valid()) {
        m_scene_color_ref = m_frame_graph->get_texture_ref(scene_color_handle);
    }

    wrap_mouse_pos();
}

void EditorApplication::setup_docking_layout() {
    if (ImGui::GetIO().IniFilename) {
        if (std::filesystem::exists(ImGui::GetIO().IniFilename)) {
            return;
        }
    }

    auto id = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
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

auto EditorApplication::create_input_actions() noexcept -> void {
    m_input_actions.clear();
}

auto EditorApplication::wrap_mouse_pos() noexcept -> void {
}

auto EditorApplication::draw_scene_panel() noexcept -> void {
    ImGui::TextUnformatted(k_editor_tutorial_text);
    ImGui::Separator();

    if (ImGui::Button("Open Scene")) {
        auto path = ImGui::FileDialogue(ImGui::FileDialogueMode::Open,
                                        {"USD Files", "*.usda *.usdc *.usd", "All Files", "*"});
        if (!path.empty()) {
            auto stage = pxr::UsdStage::Open(path);
            if (stage) {
                revoke_stage_listener();
                m_world.clear();
                m_selected_object = -1;
                m_stage = stage;
                rendering::populate_from_stage(m_world, m_stage, webgpu_context()->device());
                register_stage_listener();
                log(LogLevel::Info, "Loaded scene: {} ({} objects)", path, m_world.objects.size());
            } else {
                log(LogLevel::Error, "Failed to open scene: {}", path);
            }
        }
    }
    ImGui::SameLine();
    ImGui::BeginDisabled();
    ImGui::Button("Save Scene");
    ImGui::EndDisabled();
}

auto EditorApplication::draw_object_panel() noexcept -> void {
    ImGui::Text("Objects (%zu)", m_world.objects.size());
    ImGui::Separator();

    for (int i = 0; i < static_cast<int>(m_world.objects.size()); ++i) {
        auto const& obj = m_world.objects[i];
        auto const& label = obj.prim_path.empty() ? std::to_string(i) : obj.prim_path;
        bool selected = (m_selected_object == i);
        if (ImGui::Selectable(label.c_str(), selected)) {
            m_selected_object = selected ? -1 : i;
            if (m_selected_object >= 0 && m_stage) {
                normalize_xform_ops(m_world.objects[m_selected_object].prim_path);
            }
        }
    }
}

auto EditorApplication::draw_scene_viewport() noexcept -> void {
    if (ImGui::BeginMenuBar()) {
        ImGui::TextUnformatted("Renderer: editor.renderer");
        ImGui::EndMenuBar();
    }

    auto const avail = ImGui::GetContentRegionAvail();
    auto w = static_cast<uint32_t>(avail.x > 0.0f ? avail.x : 0.0f);
    auto h = static_cast<uint32_t>(avail.y > 0.0f ? avail.y : 0.0f);

    if (w != m_viewport_width || h != m_viewport_height) {
        m_viewport_width = w;
        m_viewport_height = h;
    }

    // Track viewport screen position for ImGuizmo
    auto const cursor_pos = ImGui::GetCursorScreenPos();
    m_viewport_x = cursor_pos.x;
    m_viewport_y = cursor_pos.y;

    if (m_scene_color_ref && m_viewport_width > 0 && m_viewport_height > 0) {
        ImGui::Image(
            reinterpret_cast<ImTextureID>(m_scene_color_ref.view()),
            ImVec2(static_cast<float>(m_viewport_width), static_cast<float>(m_viewport_height)));
    } else {
        ImGui::TextUnformatted("Renderer output not available");
    }

    // ── ImGuizmo gizmo ──
    if (m_selected_object >= 0 && m_selected_object < static_cast<int>(m_world.objects.size()) &&
        m_viewport_width > 0 && m_viewport_height > 0) {
        float aspect = static_cast<float>(m_viewport_width) / static_cast<float>(m_viewport_height);
        auto view_mat = m_camera.view_matrix();
        auto proj_mat = m_camera.projection_matrix(aspect);

        auto& obj = m_world.objects[m_selected_object];

        ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());
        ImGuizmo::SetRect(m_viewport_x, m_viewport_y, static_cast<float>(m_viewport_width),
                          static_cast<float>(m_viewport_height));

        ImGuizmo::OPERATION op = ImGuizmo::TRANSLATE;
        switch (m_gizmo_op) {
            case GizmoOp::Translate:
                op = ImGuizmo::TRANSLATE;
                break;
            case GizmoOp::Rotate:
                op = ImGuizmo::ROTATE;
                break;
            case GizmoOp::Scale:
                op = ImGuizmo::SCALE;
                break;
        }

        // Use a temporary matrix so ImGuizmo doesn't directly modify RenderWorld.
        // The notice handler is the single source of truth for RenderWorld transforms.
        glm::mat4 gizmo_transform = obj.transform;
        ImGuizmo::Manipulate(glm::value_ptr(view_mat), glm::value_ptr(proj_mat), op,
                             ImGuizmo::WORLD, glm::value_ptr(gizmo_transform));

        // Write to USD stage only — the ObjectsChanged notice will update RenderWorld
        if (ImGuizmo::IsUsing() && m_stage) {
            auto prim = m_stage->GetPrimAtPath(pxr::SdfPath(obj.prim_path));
            INVARIANT_MSG(prim.IsValid(),
                          "prim_path on RenderObject must reference a valid USD prim");

            pxr::UsdGeomXformable xformable(prim);
            INVARIANT_MSG(xformable, "selected prim must be UsdGeomXformable");

            // Convert glm::mat4 (column-major) -> GfMatrix4d (row-major) via transpose
            pxr::GfMatrix4d gf_mat;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    gf_mat[r][c] = static_cast<double>(gizmo_transform[c][r]);

            // Xform ops are normalized at selection time; a single TypeTransform op
            // must exist by the time the user drags the gizmo.
            bool reset_xform_stack = false;
            auto ops = xformable.GetOrderedXformOps(&reset_xform_stack);
            INVARIANT_MSG(
                ops.size() == 1 && ops[0].GetOpType() == pxr::UsdGeomXformOp::TypeTransform,
                "xform ops must be normalized to a single TypeTransform before gizmo use");
            ops[0].Set(gf_mat);
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

auto EditorApplication::on_mouse_leave_scene_viewport() noexcept -> void {
}

auto EditorApplication::on_mouse_enter_scene_viewport() noexcept -> void {
}

auto EditorApplication::handle_input(InputEvent const& event) noexcept -> void {
    if (event.initiated_window != k_scene_view_win_name) return;

    if (event.input.input_type == InputType::KEYBOARD) {
        bool rmb_held = ImGui::IsMouseDown(ImGuiMouseButton_Right);

        // WASD/QE movement: only while right-click is held
        if (rmb_held && event.input.action_type == ActionType::HOLD) {
            float fwd = 0.0f, right = 0.0f, up = 0.0f;
            switch (event.input.key_or_button) {
                case ImGuiKey_W: fwd += 1.0f; break;
                case ImGuiKey_S: fwd -= 1.0f; break;
                case ImGuiKey_D: right += 1.0f; break;
                case ImGuiKey_A: right -= 1.0f; break;
                case ImGuiKey_E: up += 1.0f; break;
                case ImGuiKey_Q: up -= 1.0f; break;
                default: break;
            }
            if (fwd != 0.0f || right != 0.0f || up != 0.0f) {
                m_camera.move(fwd, right, up, ImGui::GetIO().DeltaTime);
            }
        }

        // Hotkeys: only on press and when right-click is NOT held
        if (!rmb_held && event.input.action_type == ActionType::PRESS) {
            switch (event.input.key_or_button) {
                case ImGuiKey_W:
                    m_gizmo_op = GizmoOp::Translate;
                    break;
                case ImGuiKey_E:
                    m_gizmo_op = GizmoOp::Rotate;
                    break;
                case ImGuiKey_R:
                    m_gizmo_op = GizmoOp::Scale;
                    break;
                case ImGuiKey_Escape:
                    m_selected_object = -1;
                    break;
                default:
                    break;
            }
        }
    }

    if (event.input.input_type == InputType::MOUSE) {
        // Left-click: pick object under cursor
        if (event.input.key_or_button == ImGuiMouseButton_Left &&
            event.input.action_type == ActionType::PRESS && !ImGuizmo::IsOver()) {
            auto local_x = event.mouse_pos.x - m_viewport_x;
            auto local_y = event.mouse_pos.y - m_viewport_y;
            if (local_x >= 0 && local_y >= 0 && local_x < static_cast<float>(m_viewport_width) &&
                local_y < static_cast<float>(m_viewport_height)) {
                m_pick_x = static_cast<uint32_t>(local_x);
                m_pick_y = static_cast<uint32_t>(local_y);
                m_pick_requested = true;
            }
        }

        // Right-click drag: orbit camera
        if (event.input.key_or_button == ImGuiMouseButton_Right &&
            event.input.action_type == ActionType::HOLD) {
            m_camera.orbit(event.normalized_mouse_delta.x, event.normalized_mouse_delta.y);
        }

        // Middle-click drag: pan camera
        if (event.input.key_or_button == ImGuiMouseButton_Middle &&
            event.input.action_type == ActionType::HOLD) {
            m_camera.pan(event.normalized_mouse_delta.x, event.normalized_mouse_delta.y);
        }

        // Scroll: zoom
        if (event.input.action_type == ActionType::SCROLL) {
            m_camera.zoom(event.mouse_scroll_delta.y);
        }
    }
}
