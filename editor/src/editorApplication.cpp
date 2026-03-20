#include "editorApplication.h"

#include <core/backgroundTask.h>
#include <core/commandLine.h>
#include <core/components/imguiComponent.h>
#include <core/components/inputComponent.h>
#include <core/diagnostics.h>
#include <core/imgui/fileDialogue.h>
#include <core/profiling.h>
#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/registry.h>
#include <core/rendering/passContext.h>
#include <core/rendering/rendererConfig.h>
#include <core/rendering/sceneLoader.h>
#include <core/rendering/scenePass.h>
#include <core/rendering/webgpuContext.h>
#include <core/rendering/windowing.h>
#include <imgui_internal.h>

#include "propertyInspector.h"
// clang-format off
#include <ImGuizmo.h>  // must follow imgui.h
// clang-format on
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <stb_image_write.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <map>
#include <stdexcept>

#include "editorResources.h"
#include "passes/editorPass.h"
#include "passes/forwardPass.h"
#include "passes/gridPass.h"
#include "passes/lobePass.h"
#include "passes/wireframePass.h"

using namespace pts;
using namespace pts::editor;

static constexpr auto k_scene_setting_win_name = "Scene Settings";
static constexpr auto k_inspector_win_name = "Inspector";
static constexpr auto k_scene_view_win_name = "Scene";
static constexpr auto k_console_win_name = "Console";
static constexpr auto k_perf_win_name = "Performance";
static constexpr auto k_console_log_buffer_size = 1024;

static const std::vector<rendering::RendererConfig> k_renderer_configs = {
    {"Forward", [](const auto& sl) { return std::make_unique<ForwardPass>(sl); }},
    {"Wireframe", [](const auto& sl) { return std::make_unique<WireframePass>(sl); }},
};

EditorApplication::EditorApplication(std::string_view name, pts::LoggingManager& logging_manager)
    : WindowedApplication{name, logging_manager},
      m_shader_loader(logging_manager.get_logger_shared("shader_loader")) {
    create_input_actions();

    m_console_log_sink =
        std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(k_console_log_buffer_size);
    get_logging_manager().add_sink(m_console_log_sink);

    log(pts::LogLevel::Info, "EditorApplication created");
}

EditorApplication::~EditorApplication() {
    m_scene_load_task.reset();  // join background thread before tearing down
    m_pending_stage.Reset();
    revoke_stage_listener();
    m_renderer_pass.reset();
    m_editor_passes.clear();
    m_world.clear();
    m_stage.Reset();
    m_input.reset();
    m_imgui.reset();
    if (m_capture_buffer) {
        wgpuBufferRelease(m_capture_buffer);
        m_capture_buffer = nullptr;
    }
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
    m_resync_paths.clear();
}

void EditorApplication::on_objects_changed(const pxr::UsdNotice::ObjectsChanged& notice) {
    for (const auto& path : notice.GetResyncedPaths()) {
        if (path == pxr::SdfPath::AbsoluteRootPath()) continue;
        m_resync_paths.push_back(path.IsPropertyPath() ? path.GetPrimPath() : path);
    }
    for (const auto& path : notice.GetChangedInfoOnlyPaths()) {
        if (!path.IsPrimPath() && !path.IsPropertyPath()) continue;
        auto prim_path = path.IsPropertyPath() ? path.GetPrimPath() : path;
        if (path.IsPropertyPath() &&
            pxr::UsdGeomXformable::IsTransformationAffectedByAttrNamed(path.GetNameToken())) {
            m_dirty_xform_paths.push_back(prim_path);
        } else {
            m_resync_paths.push_back(prim_path);
        }
    }
}

void EditorApplication::process_dirty_prims() {
    if (!m_stage) return;

    if (!m_resync_paths.empty()) {
        // Deduplicate
        std::sort(m_resync_paths.begin(), m_resync_paths.end());
        m_resync_paths.erase(std::unique(m_resync_paths.begin(), m_resync_paths.end()),
                             m_resync_paths.end());

        // Selection is prim-path based — survives resync automatically

        {
            auto scope = m_world.begin_sync();
            for (const auto& resync_path : m_resync_paths) {
                // Handle ancestor resyncs: resync children under this path
                std::vector<pxr::SdfPath> children_to_resync;
                m_world.for_each_prim([&](std::string_view path, rendering::PrimSlot) {
                    auto child = pxr::SdfPath(std::string{path});
                    if (child.HasPrefix(resync_path) && child != resync_path) {
                        children_to_resync.push_back(child);
                    }
                });

                // Sync the prim itself
                rendering::sync_prim(scope, m_stage, resync_path);

                // Sync affected children
                for (const auto& child_path : children_to_resync) {
                    rendering::sync_prim(scope, m_stage, child_path);
                }
            }
        }  // mesh_version bumped here

        m_world.upload_all_meshes(webgpu_context()->device());

        // Invalidate selection if prim was removed
        if (!m_selected_prim.IsEmpty() && !m_stage->GetPrimAtPath(m_selected_prim).IsValid()) {
            m_selected_prim = pxr::SdfPath();
        }

        log(LogLevel::Debug, "Resynced {} prim(s)", m_resync_paths.size());
        m_resync_paths.clear();
    }

    // Xform-only changes — lightweight transform update (no mesh re-upload)
    if (!m_dirty_xform_paths.empty()) {
        std::sort(m_dirty_xform_paths.begin(), m_dirty_xform_paths.end());
        m_dirty_xform_paths.erase(
            std::unique(m_dirty_xform_paths.begin(), m_dirty_xform_paths.end()),
            m_dirty_xform_paths.end());

        m_world.update_transforms(m_stage, m_dirty_xform_paths);
        m_dirty_xform_paths.clear();
    }
}

void EditorApplication::normalize_xform_ops(const std::string& prim_path) {
    PRECONDITION(m_stage);
    auto prim = m_stage->GetPrimAtPath(pxr::SdfPath(prim_path));
    INVARIANT_MSG(prim.IsValid(), "prim_path on ObjectSlot must reference a valid USD prim");

    pxr::UsdGeomXformable xformable(prim);
    if (!xformable) return;

    bool reset_xform_stack = false;
    auto ops = xformable.GetOrderedXformOps(&reset_xform_stack);
    if (ops.size() == 1 && ops[0].GetOpType() == pxr::UsdGeomXformOp::TypeTransform) return;

    pxr::GfMatrix4d xf;
    bool resetsXformStack;
    xformable.GetLocalTransformation(&xf, &resetsXformStack, pxr::UsdTimeCode::Default());
    xformable.ClearXformOpOrder();
    xformable.AddTransformOp().Set(xf);
}

pxr::SdfPath EditorApplication::find_unique_prim_path(std::string_view base_name) {
    PRECONDITION(m_stage);
    PRECONDITION(!base_name.empty());

    static const auto k_root = pxr::SdfPath("/Root");

    if (!m_stage->GetPrimAtPath(k_root).IsValid()) {
        pxr::UsdGeomXform::Define(m_stage, k_root);
    }

    auto candidate = k_root.AppendChild(pxr::TfToken(std::string(base_name)));
    if (!m_stage->GetPrimAtPath(candidate).IsValid()) {
        return candidate;
    }

    for (int i = 1;; ++i) {
        candidate =
            k_root.AppendChild(pxr::TfToken(std::string(base_name) + "_" + std::to_string(i)));
        if (!m_stage->GetPrimAtPath(candidate).IsValid()) {
            return candidate;
        }
    }
}

void EditorApplication::ensure_default_light() {
    if (!m_stage) return;

    // Check if the stage already has any lights
    bool has_light = false;
    m_world.for_each_prim([&](std::string_view, rendering::PrimSlot slot) {
        if (slot.kind == rendering::PrimSlot::Kind::Light) has_light = true;
    });
    if (has_light) return;

    auto path = find_unique_prim_path("DomeLight");
    auto light = pxr::UsdLuxDomeLight::Define(m_stage, path);
    light.GetIntensityAttr().Set(2.0f);

    // Sync the new light into the world (listener isn't registered yet)
    auto scope = m_world.begin_sync();
    rendering::sync_prim(scope, m_stage, path);
}

void EditorApplication::register_args(CommandLine& cli) {
    WindowedApplication::register_args(cli);
    cli.add_string("capture-and-quit", "Render, capture viewport to PNG, then quit", std::nullopt);
    cli.add_string("usd", "Load USD file instead of embedded default scene", std::nullopt);
    cli.add_string("usd-override", "Apply override layer on top of loaded scene", std::nullopt);
    cli.add_int("frames", "Frames to render before capture", 1);
    cli.add_string("renderer", "Select renderer by name (e.g. Forward, Wireframe)", std::nullopt);
    cli.add_string("debug-output", "Capture debug target instead of scene_color", std::nullopt);
}

void EditorApplication::process_args(const CommandLine& cli) {
    WindowedApplication::process_args(cli);

    if (cli.has("capture-and-quit")) {
        auto path = cli.get_string("capture-and-quit");
        if (path.empty()) {
            // Generate default path: _captures/<timestamp>.png
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            char buf[64];
            std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&time_t));
            std::filesystem::create_directories("_captures");
            path = "_captures/" + std::string(buf) + ".png";
        }
        m_app_config.capture_output = std::move(path);
    }

    if (cli.has("usd")) {
        m_app_config.usd_path = cli.get_string("usd");
    }
    if (cli.has("usd-override")) {
        m_app_config.usd_override_path = cli.get_string("usd-override");
    }
    if (cli.has("frames")) {
        m_app_config.capture_frames = cli.get_int("frames", 1);
        PRECONDITION_MSG(m_app_config.capture_frames >= 1, "frames must be >= 1");
    }
    if (cli.has("renderer")) {
        m_app_config.renderer_name = cli.get_string("renderer");
    }
    if (cli.has("debug-output")) {
        m_app_config.debug_output_name = cli.get_string("debug-output");
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

    // Load scene
    if (!m_app_config.usd_path.empty()) {
        // Load USD file from disk
        m_stage = pxr::UsdStage::Open(m_app_config.usd_path);
        INVARIANT_MSG(m_stage, "Failed to open USD stage from path");
        rendering::populate_from_stage(m_world, m_stage);
        m_world.upload_all_meshes(device);
        ensure_default_light();
        register_stage_listener();
        log(LogLevel::Info, "Loaded scene from {} ({} objects)", m_app_config.usd_path,
            m_world.get_objects().size());
    } else {
        // Load embedded default scene
        auto usda = editor_resources::get_resource("assets/scenes/primitives.usda");
        if (usda) {
            auto layer = pxr::SdfLayer::CreateAnonymous(".usda");
            layer->ImportFromString(std::string{*usda});
            m_stage = pxr::UsdStage::Open(layer);
            rendering::populate_from_stage(m_world, m_stage);
            m_world.upload_all_meshes(device);
            ensure_default_light();
            register_stage_listener();
            log(LogLevel::Info, "Loaded default scene ({} objects)", m_world.get_objects().size());
        } else {
            log(LogLevel::Warning, "Missing embedded resource: assets/scenes/primitives.usda");
        }
    }

    // Apply USD override layer
    if (!m_app_config.usd_override_path.empty()) {
        INVARIANT_MSG(m_stage, "Cannot apply USD override without a loaded stage");
        auto session = m_stage->GetSessionLayer();
        session->InsertSubLayerPath(m_app_config.usd_override_path);
        // Re-sync world from stage with override applied
        revoke_stage_listener();
        m_world.clear();
        rendering::populate_from_stage(m_world, m_stage);
        m_world.upload_all_meshes(device);
        ensure_default_light();
        register_stage_listener();
        log(LogLevel::Info, "Applied USD override: {}", m_app_config.usd_override_path);
    }

    // Register shaders for hot-reload
    m_shader_loader.register_shader(
        "editor/generated/shaders/forward.wgsl", "editor/shaders/forward.slang",
        "editor/generated/shaders/forward.wgsl", editor_resources::get_resource);
    m_shader_loader.register_shader(
        "editor/generated/shaders/grid.wgsl", "editor/shaders/grid.slang",
        "editor/generated/shaders/grid.wgsl", editor_resources::get_resource);
    m_shader_loader.register_shader(
        "editor/generated/shaders/picking.wgsl", "editor/shaders/picking.slang",
        "editor/generated/shaders/picking.wgsl", editor_resources::get_resource);
    m_shader_loader.register_shader(
        "editor/generated/shaders/wireframe.wgsl", "editor/shaders/wireframe.slang",
        "editor/generated/shaders/wireframe.wgsl", editor_resources::get_resource);
    m_shader_loader.register_shader(
        "editor/generated/shaders/gizmo.wgsl", "editor/shaders/gizmo.slang",
        "editor/generated/shaders/gizmo.wgsl", editor_resources::get_resource);
    m_shader_loader.register_shader(
        "editor/generated/shaders/lobe.wgsl", "editor/shaders/lobe.slang",
        "editor/generated/shaders/lobe.wgsl", editor_resources::get_resource);

    // Create editor passes (always-on, independent of renderer choice)
    {
        auto& dev = webgpu_context()->device();
        m_editor_passes.push_back(std::make_unique<GridPass>(m_shader_loader));
        m_editor_passes.push_back(std::make_unique<EditorPass>(m_shader_loader));
        m_editor_passes.push_back(std::make_unique<LobePass>(m_shader_loader));
        for (auto& p : m_editor_passes) {
            p->setup(dev);
        }
        for (auto& p : m_editor_passes) {
            if (auto* ep = dynamic_cast<EditorPass*>(p.get())) {
                m_editor_pass = ep;
            }
        }
    }

    // Set up renderer pass — optionally select by name
    if (!m_app_config.renderer_name.empty()) {
        bool found = false;
        for (size_t i = 0; i < k_renderer_configs.size(); ++i) {
            if (k_renderer_configs[i].name == m_app_config.renderer_name) {
                set_renderer_config(i);
                found = true;
                break;
            }
        }
        INVARIANT_MSG(found, "Unknown renderer name");
    } else {
        set_renderer_config(0);
    }

    // Resolve --debug-output to m_debug_target_selection
    if (!m_app_config.debug_output_name.empty()) {
        int global_index = 1;  // 1-based (0 = "Off")
        bool found = false;
        for_each_pass([&](auto& pass) {
            auto [names, count] = pass.debug_target_names();
            for (uint32_t i = 0; i < count; ++i) {
                if (names[i] == m_app_config.debug_output_name) {
                    m_debug_target_selection = global_index;
                    found = true;
                    return;
                }
                ++global_index;
            }
        });
        INVARIANT_MSG(found, "Unknown debug output name");
    }

    // Camera defaults
    m_camera.set_target({0.0f, 0.0f, 0.0f});
    m_camera.set_distance(3.0f);
    m_camera.set_fov_y(60.0f);

    // In capture mode, set fixed viewport size (no ImGui layout)
    if (m_app_config.is_capture_mode()) {
        m_viewport_width = 1280;
        m_viewport_height = 720;
    }
}

void EditorApplication::set_renderer_config(size_t index) {
    PRECONDITION(index < k_renderer_configs.size());
    m_renderer_pass = k_renderer_configs[index].factory(m_shader_loader);
    auto& device = webgpu_context()->device();
    m_renderer_pass->setup(device);
    m_active_config_index = index;
    m_debug_target_selection = 0;
    m_active_debug_ref = {};
}

void EditorApplication::update(float /*dt*/) {
    // Input polling and ImGui drawing happen in render() to ensure proper
    // synchronization with ImGui::NewFrame() and the FrameGraph.
}

void EditorApplication::render(FrameContext& ctx) {
    PTS_ZONE_SCOPED;
    if (!m_imgui) return;
    if (viewport() && viewport()->should_close()) return;

    bool const capture_mode = m_app_config.is_capture_mode();
    ++m_frame_count;

    // ── Capture readback: if buffer was copied last frame, map and write PNG ──
    if (capture_mode && m_capture_pending) {
        // Process events to complete the mapAsync
        wgpuInstanceProcessEvents(ctx.device().instance());

        auto map_state = wgpuBufferGetMapState(m_capture_buffer);
        if (map_state == WGPUBufferMapState_Mapped) {
            uint32_t const buf_size = m_capture_bytes_per_row * m_viewport_height;
            auto const* mapped = static_cast<const uint8_t*>(
                wgpuBufferGetConstMappedRange(m_capture_buffer, 0, buf_size));
            INVARIANT(mapped);

            // Copy row-by-row, stripping padding
            std::vector<uint8_t> pixels(m_viewport_width * m_viewport_height * 4);
            for (uint32_t y = 0; y < m_viewport_height; ++y) {
                std::memcpy(&pixels[y * m_viewport_width * 4], mapped + y * m_capture_bytes_per_row,
                            m_viewport_width * 4);
            }

            wgpuBufferUnmap(m_capture_buffer);
            wgpuBufferRelease(m_capture_buffer);
            m_capture_buffer = nullptr;
            m_capture_pending = false;

            // Ensure output directory exists
            auto parent = std::filesystem::path(m_app_config.capture_output).parent_path();
            if (!parent.empty()) {
                std::filesystem::create_directories(parent);
            }

            int const ok = stbi_write_png(m_app_config.capture_output.c_str(),
                                          static_cast<int>(m_viewport_width),
                                          static_cast<int>(m_viewport_height), 4, pixels.data(),
                                          static_cast<int>(m_viewport_width * 4));
            INVARIANT_MSG(ok, "stbi_write_png failed");

            log(LogLevel::Info, "Captured {}x{} to {}", m_viewport_width, m_viewport_height,
                m_app_config.capture_output);
            viewport()->request_close();
            return;
        }
        // Not mapped yet — continue rendering to pump events
    }

    // Finalize background scene load
    if (m_scene_load_task && m_scene_load_task->is_done()) {
        auto world = m_scene_load_task->take_result();
        m_scene_load_task.reset();

        // GPU upload on main thread
        world.upload_all_meshes(webgpu_context()->device());

        // Swap into editor state — invalidate stale picking from old world
        revoke_stage_listener();
        m_picking_readback = webgpu::BufferReadback{};
        m_pick_requested = false;
        m_world = std::move(world);
        m_selected_prim = pxr::SdfPath();
        m_stage = std::move(m_pending_stage);
        m_pending_stage.Reset();
        ensure_default_light();
        register_stage_listener();

        // Re-setup passes (they cache world references)
        set_renderer_config(m_active_config_index);
        auto& dev = webgpu_context()->device();
        for (auto& p : m_editor_passes) p->setup(dev);

        log(LogLevel::Info, "Loaded scene ({} objects)", m_world.get_objects().size());
    }

    // Process deferred USD change notifications before rendering
    process_dirty_prims();

#ifdef PTS_SHADER_HOT_RELOAD
    {
        auto changed = m_shader_loader.try_finish_reload();
        m_shader_loader.poll_and_start_reload();
        if (!changed.empty()) {
            auto const& device = webgpu_context()->device();
            for_each_pass([&](auto& pass) { pass.on_shaders_reloaded(device); });
        }
    }
#endif

    auto scope = m_imgui->frame_scope();
    ImGuizmo::BeginFrame();

    if (!capture_mode) {
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
            draw_inspector_panel();
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

        // Pass-owned ImGui windows
        for_each_pass([](auto& pass) { pass.draw_imgui(); });

        // Collect all passes for perf overlay
        std::vector<rendering::IScenePass*> all_passes;
        for_each_pass([&](auto& pass) { all_passes.push_back(&pass); });
        m_perf_overlay.draw(get_delta_time(), m_world, *m_frame_graph, all_passes,
                            k_renderer_configs[m_active_config_index].name, m_viewport_width,
                            m_viewport_height);

        m_loading_overlay.draw();
    }

    // ── Frame graph ──
    auto const& device = ctx.device();
    auto queue = device.queue();

    m_frame_graph->begin_frame();

    bool has_viewport = m_viewport_width > 0 && m_viewport_height > 0;

    rendering::ResourceHandle scene_color_handle;

    // Resolve selected prim to picking ID via EditorPass table
    uint32_t selected_picking_id = UINT32_MAX;
    if (!capture_mode && !m_selected_prim.IsEmpty() && m_editor_pass) {
        selected_picking_id = m_editor_pass->find_picking_id(m_selected_prim.GetString());
    }

    rendering::PassContext pass_ctx{
        device,         queue,          m_camera,   m_world, m_viewport_width,    m_viewport_height,
        glm::mat4(1.f), glm::mat4(1.f), get_time(), 0,       selected_picking_id,
    };

    if (has_viewport) {
        float aspect = static_cast<float>(m_viewport_width) / static_cast<float>(m_viewport_height);
        pass_ctx.view_matrix = m_camera.view_matrix();
        pass_ctx.proj_matrix = m_camera.projection_matrix(aspect);

        m_world.prepare_gpu_buffers(device, queue);
    }

    // In capture mode, only add the renderer pass (skip editor passes)
    if (capture_mode) {
        if (m_renderer_pass && m_renderer_pass->is_ready() &&
            !(m_renderer_pass->requires_viewport() && !has_viewport)) {
            m_renderer_pass->add_to_frame_graph(*m_frame_graph, pass_ctx);
        }
    } else {
        for_each_pass([&](auto& pass) {
            if (!pass.is_ready()) return;
            if (pass.requires_viewport() && !has_viewport) return;
            pass.add_to_frame_graph(*m_frame_graph, pass_ctx);
        });
    }

    if (has_viewport) {
        // Look up scene_color resource that passes created
        rendering::TextureDesc color_desc;
        color_desc.width = m_viewport_width;
        color_desc.height = m_viewport_height;
        color_desc.format = WGPUTextureFormat_RGBA8Unorm;
        color_desc.clear_color = {0.15, 0.15, 0.18, 1.0};
        if (capture_mode) {
            color_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                                             WGPUTextureUsage_CopySrc);
        }
        scene_color_handle = m_frame_graph->find_or_create("scene_color", color_desc);
    }

    // Declare reads on all debug target textures so frame graph tracks them
    std::vector<rendering::ResourceHandle> debug_target_handles;
    if (has_viewport) {
        rendering::TextureDesc debug_desc;
        debug_desc.width = m_viewport_width;
        debug_desc.height = m_viewport_height;
        debug_desc.format = WGPUTextureFormat_RGBA8Unorm;
        debug_desc.clear_color = {0, 0, 0, 1};
        if (capture_mode) {
            debug_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                                             WGPUTextureUsage_CopySrc);
        }

        // In capture mode, only collect debug targets from the renderer pass
        auto collect_debug_targets = [&](auto& pass) {
            auto [names, count] = pass.debug_target_names();
            for (uint32_t i = 0; i < count; ++i) {
                auto h =
                    m_frame_graph->find_or_create(std::string("debug_") + names[i], debug_desc);
                if (h.is_valid()) {
                    debug_target_handles.push_back(h);
                }
            }
        };
        if (capture_mode) {
            if (m_renderer_pass) collect_debug_targets(*m_renderer_pass);
        } else {
            for_each_pass(collect_debug_targets);
        }
    }

    if (!capture_mode) {
        // ImGui overlay pass — declare reads on any texture that ImGui::Image references
        auto imgui_builder = m_frame_graph->add_pass("imgui")
                                 .color(ctx.surface_view(), WGPUColor{0.08, 0.08, 0.12, 1.0})
                                 .present();
        if (has_viewport && scene_color_handle.is_valid()) {
            imgui_builder.read(scene_color_handle);
        }

        for (auto h : debug_target_handles) {
            imgui_builder.read(h);
        }

        // Declare read on gizmo overlay so ImGui can composite it
        rendering::ResourceHandle gizmo_overlay_handle;
        if (has_viewport) {
            rendering::TextureDesc gizmo_desc;
            gizmo_desc.width = m_viewport_width;
            gizmo_desc.height = m_viewport_height;
            gizmo_desc.format = WGPUTextureFormat_RGBA8Unorm;
            gizmo_desc.clear_color = {0, 0, 0, 0};
            gizmo_overlay_handle =
                m_frame_graph->find_or_create("editor_gizmo_overlay", gizmo_desc);
            if (gizmo_overlay_handle.is_valid()) {
                imgui_builder.read(gizmo_overlay_handle);
            }
        }

        {
            rendering::TextureDesc lobe_desc;
            lobe_desc.width = LobePass::k_texture_size;
            lobe_desc.height = LobePass::k_texture_size;
            lobe_desc.format = WGPUTextureFormat_RGBA8Unorm;
            lobe_desc.clear_color = {0.1, 0.1, 0.1, 1.0};
            auto lobe_color_handle = m_frame_graph->find_or_create("lobe_color", lobe_desc);
            if (lobe_color_handle.is_valid()) {
                imgui_builder.read(lobe_color_handle);
            }
        }
        imgui_builder.execute([&](WGPURenderPassEncoder pass) { scope.render_into(pass); });

        // Cache gizmo overlay ref
        if (has_viewport && gizmo_overlay_handle.is_valid()) {
            m_gizmo_overlay_ref = m_frame_graph->get_texture_ref(gizmo_overlay_handle);
        } else {
            m_gizmo_overlay_ref = {};
        }
    }

    m_frame_graph->compile();
    m_frame_graph->execute(ctx.encoder());

    // ── Capture: copy texture to staging buffer after target frame ──
    if (capture_mode && !m_capture_pending && m_frame_count >= m_app_config.capture_frames) {
        // Determine which texture to capture
        rendering::TextureRef capture_ref;
        if (m_debug_target_selection > 0 &&
            static_cast<size_t>(m_debug_target_selection - 1) < debug_target_handles.size()) {
            capture_ref =
                m_frame_graph->get_texture_ref(debug_target_handles[m_debug_target_selection - 1]);
        } else {
            capture_ref = m_frame_graph->get_texture_ref(scene_color_handle);
        }
        INVARIANT_MSG(capture_ref, "Capture target texture not available");

        // 256-byte row alignment required by WebGPU
        m_capture_bytes_per_row = ((m_viewport_width * 4 + 255) / 256) * 256;
        uint32_t buf_size = m_capture_bytes_per_row * m_viewport_height;

        WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        buf_desc.size = buf_size;
        buf_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        m_capture_buffer = wgpuDeviceCreateBuffer(device.handle(), &buf_desc);
        INVARIANT(m_capture_buffer);

        WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        src.texture = capture_ref.texture();
        src.mipLevel = 0;
        src.origin = {0, 0, 0};

        WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
        dst.buffer = m_capture_buffer;
        dst.layout.offset = 0;
        dst.layout.bytesPerRow = m_capture_bytes_per_row;
        dst.layout.rowsPerImage = m_viewport_height;

        WGPUExtent3D extent = {m_viewport_width, m_viewport_height, 1};
        wgpuCommandEncoderCopyTextureToBuffer(ctx.encoder(), &src, &dst, &extent);

        // Issue mapAsync — will be checked next frame
        WGPUBufferMapCallbackInfo map_cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        map_cb.mode = WGPUCallbackMode_AllowProcessEvents;
        map_cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void*, void*) {
            INVARIANT_MSG(status == WGPUMapAsyncStatus_Success, "Capture buffer mapAsync failed");
        };
        wgpuBufferMapAsync(m_capture_buffer, WGPUMapMode_Read, 0, buf_size, map_cb);
        m_capture_pending = true;
    }

    // ── GPU picking readback ──
    m_picking_readback.tick();

    if (auto picked_id = m_picking_readback.try_read_u32()) {
        if (*picked_id == UINT32_MAX) {
            m_selected_prim = pxr::SdfPath();
        } else if (m_editor_pass) {
            auto path = m_editor_pass->resolve_picking_id(*picked_id);
            if (!path.empty()) {
                m_selected_prim = pxr::SdfPath(std::string(path));
                if (m_stage) {
                    normalize_xform_ops(std::string(path));
                }
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

    if (!capture_mode) {
        // Store scene color ref for next frame's ImGui::Image
        if (has_viewport && scene_color_handle.is_valid()) {
            m_scene_color_ref = m_frame_graph->get_texture_ref(scene_color_handle);
        }

        // Cache the active debug target ref (selection 1 maps to debug_target_handles[0])
        if (m_debug_target_selection > 0 &&
            static_cast<size_t>(m_debug_target_selection - 1) < debug_target_handles.size()) {
            m_active_debug_ref =
                m_frame_graph->get_texture_ref(debug_target_handles[m_debug_target_selection - 1]);
        } else {
            m_active_debug_ref = {};
        }

        // Let passes cache their texture refs for ImGui display
        for_each_pass([&](auto& pass) { pass.update_texture_refs(*m_frame_graph); });

        wrap_mouse_pos();
    }

    PTS_FRAME_MARK;
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
    ImGui::DockBuilderDockWindow(k_perf_win_name, down);
    ImGui::DockBuilderDockWindow("BRDF Lobe", down);
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
        ImGui::FileDialogueAsync(
            ImGui::FileDialogueMode::Open, ".usda,.usdc,.usd",
            [this](ImGui::FileDialogueResult result) {
                auto layer = pxr::SdfLayer::CreateAnonymous(result.name);
                if (!layer || !layer->ImportFromString(result.contents)) {
                    log(LogLevel::Error, "Failed to parse scene: {}", result.name);
                    return;
                }
                auto stage = pxr::UsdStage::Open(layer);
                if (!stage) {
                    log(LogLevel::Error, "Failed to open stage: {}", result.name);
                    return;
                }

                // Cancel any in-flight load
                m_scene_load_task.reset();
                m_pending_stage.Reset();

                // Store stage for later (background thread will read it)
                m_pending_stage = stage;

                // Kick off background CPU extraction
                m_scene_load_task = std::make_unique<BackgroundTask<rendering::RenderWorld>>(
                    "Loading Scene", [stage](TaskProgress& progress) -> rendering::RenderWorld {
                        return rendering::populate_from_stage(stage, progress);
                    });

                // Track in overlay
                m_loading_overlay.track({
                    "Loading Scene",
                    [this] { return !m_scene_load_task || m_scene_load_task->is_done(); },
                    [this] { return m_scene_load_task ? m_scene_load_task->progress() : 1.0f; },
                    [this] {
                        return m_scene_load_task ? m_scene_load_task->status() : std::string{};
                    },
                });

                log(LogLevel::Info, "Loading scene: {} (background)", result.name);
            });
    }

    ImGui::SameLine();
    if (m_stage && ImGui::Button("Add")) {
        ImGui::OpenPopup("AddPrimPopup");
    }
    if (ImGui::BeginPopup("AddPrimPopup")) {
        std::map<std::string, std::vector<const rendering::PrimFactory*>> grouped;
        for (auto* adapter : rendering::k_scene_adapters()) {
            auto factories = adapter->get_factories();
            for (auto& f : factories) {
                grouped[f.category].push_back(nullptr);
            }
        }
        // Re-collect with stable pointers via a local vector
        std::vector<rendering::PrimFactory> all_factories;
        for (auto* adapter : rendering::k_scene_adapters()) {
            auto factories = adapter->get_factories();
            all_factories.insert(all_factories.end(), factories.begin(), factories.end());
        }
        grouped.clear();
        for (const auto& f : all_factories) {
            grouped[f.category].push_back(&f);
        }
        for (const auto& [category, factories] : grouped) {
            if (ImGui::BeginMenu(category.c_str())) {
                for (const auto* factory : factories) {
                    if (ImGui::MenuItem(factory->display_name.c_str())) {
                        auto path = find_unique_prim_path(factory->base_name);
                        factory->define(m_stage, path);
                        normalize_xform_ops(path.GetString());
                        m_selected_prim = path;
                    }
                }
                ImGui::EndMenu();
            }
        }
        ImGui::EndPopup();
    }
}

auto EditorApplication::draw_inspector_panel() noexcept -> void {
    if (!m_stage) {
        ImGui::TextUnformatted("No stage loaded");
        return;
    }
    auto root = m_stage->GetPseudoRoot();
    for (auto const& child : root.GetChildren()) {
        draw_prim_tree(child);
    }

    ImGui::Separator();
    if (!m_selected_prim.IsEmpty()) {
        auto prim = m_stage->GetPrimAtPath(m_selected_prim);
        if (prim.IsValid()) {
            draw_prim_properties(prim);
        }
    } else {
        ImGui::TextDisabled("Select a prim to inspect properties");
    }
}

void EditorApplication::draw_prim_tree(const pxr::UsdPrim& prim) {
    auto path = prim.GetPath();
    auto name = prim.GetName().GetString();
    auto type_name = prim.GetTypeName().GetString();

    bool is_selected = (m_selected_prim == path);
    bool has_children = !prim.GetChildren().empty();

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;
    if (!has_children) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

    // Label: "Name (TypeName)" or just "Name"
    std::string label = type_name.empty() ? name : name + " (" + type_name + ")";

    // Auto-open parent nodes when a child is selected (e.g. via picking)
    if (!is_selected && !m_selected_prim.IsEmpty() && m_selected_prim.HasPrefix(path)) {
        ImGui::SetNextItemOpen(true);
    }

    bool node_open = ImGui::TreeNodeEx(path.GetText(), flags, "%s", label.c_str());

    // Auto-scroll to the selected prim in the tree
    if (is_selected) {
        ImGui::ScrollToItem();
    }

    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        if (is_selected) {
            m_selected_prim = pxr::SdfPath();
        } else {
            m_selected_prim = path;
            if (pxr::UsdGeomXformable xformable{prim}; xformable) {
                normalize_xform_ops(path.GetString());
            }
        }
    }

    if (node_open && has_children) {
        for (auto const& child : prim.GetChildren()) {
            draw_prim_tree(child);
        }
        ImGui::TreePop();
    }
}

auto EditorApplication::draw_scene_viewport() noexcept -> void {
    if (ImGui::BeginMenuBar()) {
        ImGui::Text("Renderer:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        if (ImGui::BeginCombo("##renderer",
                              k_renderer_configs[m_active_config_index].name.c_str())) {
            for (size_t i = 0; i < k_renderer_configs.size(); ++i) {
                bool selected = (i == m_active_config_index);
                if (ImGui::Selectable(k_renderer_configs[i].name.c_str(), selected)) {
                    if (i != m_active_config_index) {
                        set_renderer_config(i);
                    }
                }
            }
            ImGui::EndCombo();
        }
        // Debug target dropdown
        ImGui::SameLine();
        ImGui::Text("Debug:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(140);
        {
            // Build flat list of debug target labels from all passes
            std::vector<std::string> debug_labels;
            debug_labels.emplace_back("Off");
            for_each_pass([&](auto& pass) {
                auto [names, count] = pass.debug_target_names();
                for (uint32_t i = 0; i < count; ++i) {
                    debug_labels.emplace_back(std::string(pass.name()) + ": " + names[i]);
                }
            });
            if (m_debug_target_selection >= static_cast<int>(debug_labels.size())) {
                m_debug_target_selection = 0;
            }
            if (ImGui::BeginCombo("##debug", debug_labels[m_debug_target_selection].c_str())) {
                for (int i = 0; i < static_cast<int>(debug_labels.size()); ++i) {
                    bool selected = (i == m_debug_target_selection);
                    if (ImGui::Selectable(debug_labels[i].c_str(), selected)) {
                        m_debug_target_selection = i;
                    }
                }
                ImGui::EndCombo();
            }
        }
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

    {
        auto& display_ref = (m_debug_target_selection > 0 && m_active_debug_ref)
                                ? m_active_debug_ref
                                : m_scene_color_ref;
        if (display_ref && m_viewport_width > 0 && m_viewport_height > 0) {
            ImGui::Image(reinterpret_cast<ImTextureID>(display_ref.view()),
                         ImVec2(static_cast<float>(m_viewport_width),
                                static_cast<float>(m_viewport_height)));
            // Overlay gizmo wireframes on top (visible in all views including debug)
            if (m_gizmo_overlay_ref) {
                auto* draw_list = ImGui::GetWindowDrawList();
                ImVec2 p_min(m_viewport_x, m_viewport_y);
                ImVec2 p_max(m_viewport_x + static_cast<float>(m_viewport_width),
                             m_viewport_y + static_cast<float>(m_viewport_height));
                draw_list->AddImage(reinterpret_cast<ImTextureID>(m_gizmo_overlay_ref.view()),
                                    p_min, p_max);
            }
        } else {
            ImGui::TextUnformatted("Renderer output not available");
        }
    }

    // ── ImGuizmo gizmo ──
    if (!m_selected_prim.IsEmpty() && m_stage && m_viewport_width > 0 && m_viewport_height > 0) {
        auto prim = m_stage->GetPrimAtPath(m_selected_prim);
        pxr::UsdGeomXformable xformable(prim);
        if (prim.IsValid() && xformable) {
            float aspect =
                static_cast<float>(m_viewport_width) / static_cast<float>(m_viewport_height);
            auto view_mat = m_camera.view_matrix();
            auto proj_mat = m_camera.projection_matrix(aspect);

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

            auto world_xf = rendering::compute_world_transform(prim);
            glm::mat4 gizmo_transform = world_xf;
            ImGuizmo::Manipulate(glm::value_ptr(view_mat), glm::value_ptr(proj_mat), op,
                                 ImGuizmo::WORLD, glm::value_ptr(gizmo_transform));

            if (ImGuizmo::IsUsing()) {
                pxr::GfMatrix4d gf_world;
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        gf_world[i][j] = static_cast<double>(gizmo_transform[i][j]);

                pxr::GfMatrix4d parent_world;
                if (auto parent = prim.GetParent(); parent && parent != m_stage->GetPseudoRoot()) {
                    parent_world = pxr::UsdGeomXformable(parent).ComputeLocalToWorldTransform(
                        pxr::UsdTimeCode::Default());
                } else {
                    parent_world.SetIdentity();
                }
                pxr::GfMatrix4d local_mat = gf_world * parent_world.GetInverse();

                bool reset_xform_stack = false;
                auto ops = xformable.GetOrderedXformOps(&reset_xform_stack);
                INVARIANT_MSG(
                    ops.size() == 1 && ops[0].GetOpType() == pxr::UsdGeomXformOp::TypeTransform,
                    "xform ops must be normalized to a single TypeTransform before gizmo use");
                ops[0].Set(local_mat);
            }
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
    bool rmb_held = ImGui::IsMouseDown(ImGuiMouseButton_Right);

    // WASD fly camera works globally while RMB is held (cursor may drift over other panels)
    if (rmb_held && event.input.input_type == InputType::KEYBOARD &&
        event.input.action_type == ActionType::HOLD) {
        float fwd = 0.0f, right = 0.0f, up = 0.0f;
        switch (event.input.key_or_button) {
            case ImGuiKey_W:
                fwd += 1.0f;
                break;
            case ImGuiKey_S:
                fwd -= 1.0f;
                break;
            case ImGuiKey_D:
                right += 1.0f;
                break;
            case ImGuiKey_A:
                right -= 1.0f;
                break;
            case ImGuiKey_E:
                up += 1.0f;
                break;
            case ImGuiKey_Q:
                up -= 1.0f;
                break;
            default:
                break;
        }
        if (fwd != 0.0f || right != 0.0f || up != 0.0f) {
            m_camera.move(fwd, right, up, ImGui::GetIO().DeltaTime);
        }
        return;
    }

    if (event.initiated_window != k_scene_view_win_name) return;

    if (event.input.input_type == InputType::KEYBOARD) {
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
                    m_selected_prim = pxr::SdfPath();
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
