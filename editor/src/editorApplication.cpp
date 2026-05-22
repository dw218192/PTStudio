#include "editorApplication.h"

#include <core/commandLine.h>
#include <core/components/imguiComponent.h>
#include <core/components/inputComponent.h>
#include <core/diagnostics.h>
#include <core/imgui/fileDialogue.h>
#include <core/profiling.h>
#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/registry.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderer.h>
#include <core/rendering/rendererRegistry.h>
#include <core/rendering/sceneLoader.h>
#include <core/rendering/stageSave.h>
#include <core/rendering/webgpuContext.h>
#include <core/rendering/windowing.h>
#include <core/worker.h>
#include <imgui_internal.h>

#include "propertyInspector.h"
#include "transformDecompose.h"
// clang-format off
#include <ImGuizmo.h>  // must follow imgui.h
// clang-format on
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/usd/stage.h>
#if defined(__EMSCRIPTEN__)
#include <emscripten.h>

#include <fstream>
#endif
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <stb_image_write.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "editorResources.h"
#include "passes/editorPass.h"
#include "passes/gridPass.h"
#include "passes/lobePass.h"

using namespace pts;
using namespace pts::editor;

static constexpr auto k_scene_setting_win_name = "Scene Settings";
static constexpr auto k_inspector_win_name = "Inspector";
static constexpr auto k_scene_view_win_name = "Scene";
static constexpr auto k_console_win_name = "Console";
static constexpr auto k_perf_win_name = "Performance";
static constexpr auto k_console_log_buffer_size = 1024;

static constexpr auto k_default_renderer_name = "Forward";

static constexpr auto k_demo_scenes_dir = "assets/scenes";

static std::string display_name_from_path(const std::filesystem::path& p) {
    auto stem = p.stem().string();
    for (auto& c : stem)
        if (c == '_') c = ' ';
    if (!stem.empty())
        stem[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(stem[0])));
    return stem;
}

static void discover_demo_scenes(std::vector<std::string>& paths, std::vector<std::string>& names) {
    paths.clear();
    names.clear();
    std::error_code ec;
    for (auto& entry : std::filesystem::directory_iterator(k_demo_scenes_dir, ec)) {
        if (entry.path().extension() == ".usdz") {
            paths.push_back(entry.path().string());
            names.push_back(display_name_from_path(entry.path()));
        }
    }
    // Sort for stable dropdown order
    std::vector<size_t> indices(paths.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return names[a] < names[b]; });
    auto sorted_paths = paths;
    auto sorted_names = names;
    for (size_t i = 0; i < indices.size(); ++i) {
        paths[i] = sorted_paths[indices[i]];
        names[i] = sorted_names[indices[i]];
    }
}

EditorApplication::EditorApplication(std::string_view name, pts::LoggingManager& logging_manager)
    : GpuApplication{name, logging_manager},
      m_prep_worker(std::make_unique<Worker<CpuPrepJob, rendering::PreparedSceneData>>(
          [this](CpuPrepJob&&, TaskProgress&) -> rendering::PreparedSceneData {
              return m_world.prepare_scene_data();
          })),
      m_shader_loader(logging_manager.get_logger_shared("shader_loader")) {
    create_input_actions();

    m_console_log_sink =
        std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(k_console_log_buffer_size);
    get_logging_manager().add_sink(m_console_log_sink);

    log(pts::LogLevel::Info, "EditorApplication created");
}

EditorApplication::~EditorApplication() {
    m_prep_worker.reset();      // stop worker before tearing down world
    m_scene_load_task.reset();  // join background thread before tearing down
    m_pending_stage.Reset();
    revoke_stage_listener();
    m_renderer_pass.reset();
    m_grid_pass.reset();
    m_editor_pass.reset();
    m_lobe_pass.reset();
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
    PTS_ZONE_SCOPED;
    if (!m_stage) return;

    if (!m_resync_paths.empty()) {
        // Deduplicate
        std::sort(m_resync_paths.begin(), m_resync_paths.end());
        m_resync_paths.erase(std::unique(m_resync_paths.begin(), m_resync_paths.end()),
                             m_resync_paths.end());

        // Selection is prim-path based -- survives resync automatically

        {
            auto scope = m_world.begin_sync();
            for (const auto& resync_path : m_resync_paths) {
                // Handle ancestor resyncs: resync children under this path
                std::vector<pxr::SdfPath> children_to_resync;
                m_world.for_each_prim([&](const pxr::SdfPath& path, rendering::PrimSlot) {
                    if (path.HasPrefix(resync_path) && path != resync_path) {
                        children_to_resync.push_back(path);
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

    // Xform-only changes -- lightweight transform update (no mesh re-upload)
    if (!m_dirty_xform_paths.empty()) {
        std::sort(m_dirty_xform_paths.begin(), m_dirty_xform_paths.end());
        m_dirty_xform_paths.erase(
            std::unique(m_dirty_xform_paths.begin(), m_dirty_xform_paths.end()),
            m_dirty_xform_paths.end());

        m_world.update_transforms(m_stage, m_dirty_xform_paths);
        m_dirty_xform_paths.clear();
    }
}

void EditorApplication::normalize_xform_ops(const pxr::SdfPath& prim_path) {
    PRECONDITION(m_stage);
    auto prim = m_stage->GetPrimAtPath(prim_path);
    INVARIANT_MSG(prim.IsValid(), "prim_path on ObjectData must reference a valid USD prim");

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

pxr::SdfPath EditorApplication::find_unique_prim_path(std::string_view base_name,
                                                      const pxr::SdfPath* parent) {
    PRECONDITION(m_stage);
    PRECONDITION(!base_name.empty());

    static const auto k_root = pxr::SdfPath("/Root");

    auto parent_path = parent ? *parent : k_root;

    if (!m_stage->GetPrimAtPath(parent_path).IsValid()) {
        pxr::UsdGeomXform::Define(m_stage, parent_path);
    }

    auto candidate = parent_path.AppendChild(pxr::TfToken(std::string(base_name)));
    if (!m_stage->GetPrimAtPath(candidate).IsValid()) {
        return candidate;
    }

    for (int i = 1;; ++i) {
        candidate =
            parent_path.AppendChild(pxr::TfToken(std::string(base_name) + "_" + std::to_string(i)));
        if (!m_stage->GetPrimAtPath(candidate).IsValid()) {
            return candidate;
        }
    }
}

void EditorApplication::ensure_default_light() {
    if (!m_stage) return;

    // Check if the stage already has any lights
    bool has_light = false;
    m_world.for_each_prim([&](const pxr::SdfPath&, rendering::PrimSlot slot) {
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
    GpuApplication::register_args(cli);
    cli.add_string("capture-and-quit", "Render, capture viewport to PNG, then quit", std::nullopt,
                   std::string(""));
    cli.add_string("usd", "Load USD file instead of embedded default scene", std::nullopt);
    cli.add_string("usd-override", "Apply override layer on top of loaded scene", std::nullopt);
    cli.add_int("frames", "Frames to render before capture", 1);
    cli.add_string("renderer", "Select renderer by name (e.g. Forward, Wireframe)", std::nullopt);
    cli.add_string("debug-output", "Capture debug target instead of scene_color", std::nullopt);
    cli.add_string("camera-target", "Camera target position as x,y,z", std::nullopt);
    cli.add_string("camera-distance", "Camera orbit distance", std::nullopt);
    cli.add_string("camera-yaw", "Camera yaw in degrees", std::nullopt);
    cli.add_string("camera-pitch", "Camera pitch in degrees", std::nullopt);
    cli.add_string("camera-fov", "Camera vertical FOV in degrees", std::nullopt);
    cli.add_string("camera", "Select a scene camera by prim name", std::nullopt);
}

auto EditorApplication::process_args(const CommandLine& cli) -> ErrorCode {
    auto ec = GpuApplication::process_args(cli);
    if (ec != ErrorCode::Ok) return ec;

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
        if (!std::filesystem::exists(m_app_config.usd_path)) {
            log(LogLevel::Error, "USD file not found: {}", m_app_config.usd_path);
            return ErrorCode::InvalidArgument;
        }
    }
    if (cli.has("usd-override")) {
        m_app_config.usd_override_path = cli.get_string("usd-override");
    }
    if (cli.has("frames")) {
        m_app_config.capture_frames = cli.get_int("frames", 1);
        if (m_app_config.capture_frames < 1) {
            log(LogLevel::Error, "--frames must be >= 1 (got: {})", m_app_config.capture_frames);
            return ErrorCode::InvalidArgument;
        }
    }
    if (cli.has("renderer")) {
        m_app_config.renderer_name = cli.get_string("renderer");
    }
    if (cli.has("debug-output")) {
        m_app_config.debug_output_name = cli.get_string("debug-output");
    }
    if (cli.has("camera-target")) {
        m_app_config.camera_target = cli.get_string("camera-target");
        float x, y, z;
        if (std::sscanf(m_app_config.camera_target.c_str(), "%f,%f,%f", &x, &y, &z) != 3) {
            log(LogLevel::Error, "--camera-target must be x,y,z (got: {})",
                m_app_config.camera_target);
            return ErrorCode::InvalidArgument;
        }
    }
    if (cli.has("camera-distance")) {
        m_app_config.camera_distance = cli.get_string("camera-distance");
    }
    if (cli.has("camera-yaw")) {
        m_app_config.camera_yaw = cli.get_string("camera-yaw");
    }
    if (cli.has("camera-pitch")) {
        m_app_config.camera_pitch = cli.get_string("camera-pitch");
    }
    if (cli.has("camera-fov")) {
        m_app_config.camera_fov = cli.get_string("camera-fov");
    }
    if (cli.has("camera")) {
        m_app_config.camera_prim_path = cli.get_string("camera");
    }
    return ErrorCode::Ok;
}

void EditorApplication::on_ready() {
    // Create windowing + surface + UI components only in interactive mode
    if (!m_app_config.is_capture_mode()) {
        init_windowing();
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
    }

    auto const& device = webgpu_context()->device();

    // -- Rendering init --

    // Shader compiler -- wraps ShaderLoader so native hot-reload keeps working.
    // Sub-ticket B replaces the native branch with a SlangCompiler.
    m_shader_compiler = rendering::make_shader_compiler(m_shader_loader);

    // Frame graph
    m_frame_graph = std::make_unique<rendering::FrameGraph>(
        device, get_logging_manager().get_logger_shared("frame_graph"), m_shader_compiler.get());

    // Load scene via unified load_stage()
    discover_demo_scenes(m_demo_scene_paths, m_demo_scene_names);

    if (!m_app_config.usd_path.empty()) {
        auto stage = pxr::UsdStage::Open(m_app_config.usd_path);
        if (!stage) {
            log(LogLevel::Error, "Failed to open USD stage: {}", m_app_config.usd_path);
            request_stop();
            return;
        }
        load_stage(stage, m_app_config.usd_path);
    } else {
        INVARIANT_MSG(!m_demo_scene_paths.empty(), "No demo scenes found in assets/scenes/");
        auto stage = pxr::UsdStage::Open(m_demo_scene_paths[0]);
        INVARIANT_MSG(stage, "Failed to open default demo scene");
        load_stage(stage, m_demo_scene_names[0]);
    }

    // Register shaders for hot-reload
    m_shader_loader.register_shader(
        "renderers/forward/generated/shaders/forward.wgsl", "renderers/forward/forward.slang",
        "renderers/forward/generated/shaders/forward.wgsl", editor_resources::get_resource);
    m_shader_loader.register_shader(
        "renderers/forward/generated/shaders/skybox.wgsl", "renderers/forward/skybox.slang",
        "renderers/forward/generated/shaders/skybox.wgsl", editor_resources::get_resource);
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
    m_shader_loader.register_shader(
        "editor/generated/shaders/tonemapping.wgsl", "editor/shaders/tonemapping.slang",
        "editor/generated/shaders/tonemapping.wgsl", editor_resources::get_resource);
    m_shader_loader.register_shader(
        "editor/generated/shaders/luminance.wgsl", "editor/shaders/luminance.slang",
        "editor/generated/shaders/luminance.wgsl", editor_resources::get_resource, {"cs_main"});
    m_shader_loader.register_shader(
        "editor/generated/shaders/pathtracer.wgsl", "renderers/pathtracer/pathtracer.slang",
        "editor/generated/shaders/pathtracer.wgsl", editor_resources::get_resource, {"cs_main"});
    m_shader_loader.register_shader(
        "editor/generated/shaders/pt_blit.wgsl", "renderers/pathtracer/pt_blit.slang",
        "editor/generated/shaders/pt_blit.wgsl", editor_resources::get_resource);

    // Register shadow shader for hot-reload (vertex-only: no fragment stage)
    m_shader_loader.register_shader("core/generated/shaders/shadow/shadow_map.wgsl",
                                    "core/shaders/shadow/shadow_map.slang",
                                    "core/generated/shaders/shadow/shadow_map.wgsl",
                                    editor_resources::get_resource, {"vs_main"});

    // Register gbuffer shader for hot-reload
    m_shader_loader.register_shader(
        "core/generated/shaders/gbuffer.wgsl", "core/shaders/gbuffer.slang",
        "core/generated/shaders/gbuffer.wgsl", editor_resources::get_resource);

    // Register SSAO shaders for hot-reload
    m_shader_loader.register_shader("core/generated/shaders/ssao.wgsl", "core/shaders/ssao.slang",
                                    "core/generated/shaders/ssao.wgsl",
                                    editor_resources::get_resource);
    m_shader_loader.register_shader(
        "core/generated/shaders/bilateral_blur.wgsl", "core/shaders/bilateral_blur.slang",
        "core/generated/shaders/bilateral_blur.wgsl", editor_resources::get_resource);

    // Register contact shadow shader for hot-reload
    m_shader_loader.register_shader(
        "core/generated/shaders/contact_shadow.wgsl", "core/shaders/contact_shadow.slang",
        "core/generated/shaders/contact_shadow.wgsl", editor_resources::get_resource);

    // Register shadow-visibility gen/resolve shaders for hot-reload
    m_shader_loader.register_shader("core/generated/shaders/shadow/shadow_visibility.wgsl",
                                    "core/shaders/shadow/shadow_visibility.slang",
                                    "core/generated/shaders/shadow/shadow_visibility.wgsl",
                                    editor_resources::get_resource);
    m_shader_loader.register_shader("core/generated/shaders/shadow/temporal_resolve.wgsl",
                                    "core/shaders/shadow/temporal_resolve.slang",
                                    "core/generated/shaders/shadow/temporal_resolve.wgsl",
                                    editor_resources::get_resource);

    // Create editor passes (always-on, independent of renderer choice).
    // Resources (BGLs, pipelines, shaders) are created lazily on the first
    // render() call via the FrameGraph caches -- no eager setup step.
    {
        auto& dev = webgpu_context()->device();
        m_grid_pass = std::make_unique<GridPass>(m_shader_loader);
        m_grid_pass->ensure_initialized(dev);
        m_editor_pass = std::make_unique<EditorPass>(m_shader_loader);
        m_editor_pass->ensure_initialized(dev);
        m_lobe_pass = std::make_unique<LobePass>(m_shader_loader);
        m_lobe_pass->ensure_initialized(dev);
    }

    // Set up renderer pass -- optionally select by name
    {
        auto& entries = rendering::RendererRegistry::entries();
        INVARIANT_MSG(!entries.empty(), "No renderers registered");
        auto target_name = m_app_config.renderer_name.empty() ? k_default_renderer_name
                                                              : m_app_config.renderer_name;
        bool found = false;
        for (size_t i = 0; i < entries.size(); ++i) {
            if (entries[i].name == target_name) {
                create_renderer(i);
                found = true;
                break;
            }
        }
        if (!found) {
            log(LogLevel::Error, "Unknown renderer: {}", target_name);
            request_stop();
            return;
        }
    }

    // Resolve --debug-output to m_debug_target_selection
    if (!m_app_config.debug_output_name.empty()) {
        int global_index = 1;  // 1-based (0 = "Off")
        bool found = false;
        for_each_pass([&](auto& pass) {
            auto [targets, count] = pass.effective_debug_targets();
            for (uint32_t i = 0; i < count; ++i) {
                if (targets[i].label == m_app_config.debug_output_name) {
                    m_debug_target_selection = global_index;
                    found = true;
                    return;
                }
                ++global_index;
            }
        });
        if (!found) {
            log(LogLevel::Error, "Unknown debug output: {}", m_app_config.debug_output_name);
            request_stop();
            return;
        }
    }

    // Camera defaults, overridable via CLI
    m_camera.set_target({0.0f, 0.0f, 0.0f});
    m_camera.set_distance(3.0f);
    m_camera.set_fov_y(60.0f);

    if (!m_app_config.camera_target.empty()) {
        float x, y, z;
        // Format already validated in process_args -- parse is safe here
        std::sscanf(m_app_config.camera_target.c_str(), "%f,%f,%f", &x, &y, &z);
        m_camera.set_target({x, y, z});
    }
    if (!m_app_config.camera_distance.empty()) {
        m_camera.set_distance(std::stof(m_app_config.camera_distance));
    }
    if (!m_app_config.camera_yaw.empty()) {
        m_camera.set_yaw(glm::radians(std::stof(m_app_config.camera_yaw)));
    }
    if (!m_app_config.camera_pitch.empty()) {
        m_camera.set_pitch(glm::radians(std::stof(m_app_config.camera_pitch)));
    }
    if (!m_app_config.camera_fov.empty()) {
        m_camera.set_fov_y(std::stof(m_app_config.camera_fov));
    }

    // Select scene camera by prim path (from --camera CLI arg)
    if (!m_app_config.camera_prim_path.empty()) {
        int idx = m_world.find_camera_by_prim(pxr::SdfPath(m_app_config.camera_prim_path));
        if (idx >= 0) {
            m_active_camera_index = idx + 1;
        }
    }

    // In capture mode, set fixed viewport size (no ImGui layout)
    if (m_app_config.is_capture_mode()) {
        m_viewport_width = 1280;
        m_viewport_height = 720;
    }

    m_init_complete = true;
}

auto EditorApplication::compute_active_view(float aspect) const -> ActiveView {
    const auto& cameras = m_world.get_cameras();
    auto cameras_raw = cameras.span_raw();
    uint32_t cam_slot = static_cast<uint32_t>(m_active_camera_index - 1);
    if (m_active_camera_index > 0 && cam_slot < cameras_raw.size() &&
        cameras_raw[cam_slot].active) {
        const auto& cam = cameras_raw[cam_slot].value;
        glm::mat4 proj;
        if (cam.orthographic) {
            float half_h = cam.ortho_height * 0.5f;
            float half_w = half_h * aspect;
            proj = glm::ortho(-half_w, half_w, -half_h, half_h, cam.near_clip, cam.far_clip);
        } else {
            proj = glm::perspective(cam.fov_y_radians, aspect, cam.near_clip, cam.far_clip);
        }
        return {cam.view_matrix, proj, glm::vec3(glm::inverse(cam.view_matrix)[3])};
    }
    return {m_camera.view_matrix(), m_camera.projection_matrix(aspect), m_camera.position()};
}

void EditorApplication::create_renderer(size_t index) {
    auto& entries = rendering::RendererRegistry::entries();
    PRECONDITION(index < entries.size());
    m_renderer_pass = entries[index].factory(m_shader_loader);
    auto& device = webgpu_context()->device();
    m_renderer_pass->ensure_initialized(device);
    m_active_config_index = index;
    m_editor_passes_enabled = entries[index].editor_passes;
    m_debug_target_selection = 0;
    m_active_debug_view = nullptr;
    m_scene_color_view = nullptr;
    m_gizmo_overlay_view = nullptr;
}

void EditorApplication::update(float /*dt*/) {
    // Input polling and ImGui drawing happen in render() to ensure proper
    // synchronization with ImGui::NewFrame() and the FrameGraph.
}

void EditorApplication::save_capture_png(boost::span<const uint8_t> pixels, uint32_t width,
                                         uint32_t height, std::string_view path) {
#ifdef __EMSCRIPTEN__
    if (!m_app_config.is_capture_mode()) {
        // Interactive browser screenshot: trigger download via DOM
        // clang-format off
        EM_ASM({
            var width = $0;
            var height = $1;
            var dataPtr = $2;
            var fileName = UTF8ToString($3);
            var size = width * height * 4;
            var data = new Uint8Array(HEAPU8.buffer, dataPtr, size);
            var canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            var ctx = canvas.getContext('2d');
            var imageData = ctx.createImageData(width, height);
            imageData.data.set(data);
            ctx.putImageData(imageData, 0, 0);
            canvas.toBlob(function(blob) {
                var url = URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.href = url;
                a.download = fileName;
                a.click();
                URL.revokeObjectURL(url);
            }, 'image/png');
        }, width, height, pixels.data(), path.data());
        // clang-format on
        log(LogLevel::Info, "Screenshot download triggered: {}", path);
        return;
    }
#endif
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    int const ok =
        stbi_write_png(std::string(path).c_str(), static_cast<int>(width), static_cast<int>(height),
                       4, pixels.data(), static_cast<int>(width * 4));
    INVARIANT_MSG(ok, "stbi_write_png failed");
    log(LogLevel::Info, "Captured {}x{} to {}", width, height, path);
}

void EditorApplication::render(FrameContext& ctx) {
    PTS_ZONE_SCOPED;
    if (!m_frame_graph) return;
    if (viewport() && viewport()->should_close()) return;

    bool const capture_mode = m_app_config.is_capture_mode();
    if (!m_scene_load_task) ++m_frame_count;

    // -- Capture readback: tick the async state machine and save when ready --
    if (m_capture_readback.is_pending()) {
        m_capture_readback.tick();
        auto pixels = m_capture_readback.try_read();
        if (!pixels.empty()) {
            std::string path;
            if (capture_mode) {
                path = m_app_config.capture_output;
            } else {
                auto now = std::chrono::system_clock::now();
                auto tt = std::chrono::system_clock::to_time_t(now);
                char buf[64];
                std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&tt));
                path = std::string("_captures/") + buf + ".png";
            }
            save_capture_png(pixels, m_capture_width, m_capture_height, path);
            if (capture_mode) {
                if (viewport()) {
                    viewport()->request_close();
                }
                request_stop();
                return;
            }
        }
    }

    // Finalize background scene load
    if (m_scene_load_task && m_scene_load_task->is_done()) {
        auto world = m_scene_load_task->take_result();
        m_scene_load_task.reset();

        // GPU upload on main thread
        world.upload_all_meshes(webgpu_context()->device());

        // Quiesce the prep worker -- it captures m_world by reference, so
        // swapping the world while a job is in flight is a data race.
        m_prep_worker.reset();
        m_world = std::move(world);
        m_prep_worker = std::make_unique<Worker<CpuPrepJob, rendering::PreparedSceneData>>(
            [this](CpuPrepJob&&, TaskProgress&) -> rendering::PreparedSceneData {
                return m_world.prepare_scene_data();
            });
        m_first_prep = true;
        m_active_camera_index = 0;
        m_stage = std::move(m_pending_stage);
        m_pending_stage.Reset();
        activate_stage();

        // Re-setup passes (pass data cache lives in the world, so it was
        // already destroyed when m_world was replaced above)
        create_renderer(m_active_config_index);
        auto& dev = webgpu_context()->device();
        if (m_grid_pass) m_grid_pass->ensure_initialized(dev);
        if (m_editor_pass) m_editor_pass->ensure_initialized(dev);
        if (m_lobe_pass) m_lobe_pass->ensure_initialized(dev);

        log(LogLevel::Info, "Loaded scene ({} objects)", m_world.get_objects().size());
    }

    // Process deferred USD change notifications before rendering
    process_dirty_prims();

    // Hot-reload: ask the compiler for any sources dirty since last poll. The
    // compiler bumps its per-source revision; FrameGraph's DepTrackedSlotMap
    // drops stale shader modules on the next shader()/shader_variant() call,
    // and pipelines rebuild via their shader_module_version dep.
    if (m_shader_compiler) {
        auto dirty = m_shader_compiler->poll_dirty();
        for (const auto& key : dirty) {
            m_frame_graph->invalidate_shader(key);
            // Also invalidate the NO_DEBUG_TARGETS variant cache key, which is
            // keyed separately in the FG shader cache but shares the same
            // libslang source.
            auto dot = key.rfind('.');
            if (dot != std::string::npos) {
                m_frame_graph->invalidate_shader(key.substr(0, dot) + "_no_debug" +
                                                 key.substr(dot));
            }
        }
    }

    // Begin ImGui frame if available (interactive mode only)
    if (m_imgui) {
        m_imgui->begin_frame();
        ImGuizmo::BeginFrame();
    }

    if (m_imgui && !capture_mode) {
        // Poll input -- prev_hovered_widget makes this order-independent from UI drawing
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

        // Renderer settings window -- one shared window, each pass draws a section
        if (ImGui::Begin("Renderer")) {
            for_each_pass([](auto& pass) { pass.draw_imgui(); });
        }
        ImGui::End();

        // Collect all passes for perf overlay
        std::vector<rendering::IPass*> all_passes;
        for_each_pass([&](auto& pass) { all_passes.push_back(&pass); });
        m_perf_overlay.draw(get_delta_time(), m_world, *m_frame_graph, all_passes,
                            rendering::RendererRegistry::entries()[m_active_config_index].name,
                            m_viewport_width, m_viewport_height);

        m_loading_overlay.draw();
    }

    // -- Frame graph --
    auto const& device = ctx.device();
    auto queue = device.queue();

    m_frame_graph->begin_frame();

    bool has_viewport = m_viewport_width > 0 && m_viewport_height > 0;

    rendering::TextureDeclHandle display_color_decl;  // tone-mapped output for ImGui display
    rendering::TextureDeclHandle gizmo_overlay_decl;

    // Resolve selected prim to picking ID via EditorPass table
    uint32_t selected_picking_id = UINT32_MAX;
    if (!capture_mode && !m_selected_prim.IsEmpty() && m_editor_pass) {
        selected_picking_id = m_editor_pass->find_picking_id(m_selected_prim);
    }

    rendering::PassContext pass_ctx{
        device,
        queue,
        m_camera,
        m_world,
        m_viewport_width,
        m_viewport_height,
        glm::mat4(1.f),
        glm::mat4(1.f),
        glm::vec3(0.f),
        get_time(),
        0,
        selected_picking_id,
        m_stage_settings.meters_per_unit,
        m_stage_settings.up_axis,
    };

    if (has_viewport) {
        float aspect = static_cast<float>(m_viewport_width) / static_cast<float>(m_viewport_height);
        auto view = compute_active_view(aspect);
        pass_ctx.view_matrix = view.view_matrix;
        pass_ctx.proj_matrix = view.proj_matrix;
        pass_ctx.camera_position = view.camera_position;

        auto ibl_sampler =
            m_frame_graph->sampler(WGPUSamplerBindingType_Filtering, WGPUAddressMode_ClampToEdge,
                                   WGPUMipmapFilterMode_Linear);
        m_world.update_ibl(device, queue, ibl_sampler, m_stage_settings.up_axis);

        if (capture_mode) {
            // Capture mode: always synchronous for deterministic output
            m_world.prepare_gpu_buffers(device, queue);
        } else if (m_first_prep) {
            // First frame: synchronous fallback (no stale data)
            m_world.prepare_gpu_buffers(device, queue);
            m_prep_worker->take_result();  // discard stale result after scene load
            m_prep_worker->submit(CpuPrepJob{});
            m_first_prep = false;
        } else {
            // Async: upload previous frame's result, submit next frame's work
            if (m_prep_worker->has_result()) {
                auto prepared = m_prep_worker->take_result();
                INVARIANT(prepared.has_value());
                m_world.upload_prepared_data(device, queue, std::move(*prepared));
            }
            m_prep_worker->submit(CpuPrepJob{});
        }
    }

    // 1. Renderer produces display-ready color (includes tone mapping)
    rendering::TextureDeclHandle scene_color_decl;
    rendering::TextureDeclHandle scene_depth_decl;
    {
        PTS_ZONE_NAMED("add_to_frame_graph");
        if (m_renderer_pass && !(m_renderer_pass->requires_viewport() && !has_viewport)) {
            auto out = m_renderer_pass->add_to_frame_graph(*m_frame_graph, pass_ctx);
            display_color_decl = out.color;
            scene_color_decl = out.hdr_color;
            scene_depth_decl = out.depth;
        }

        // 2. Editor overlays (called explicitly, not through virtual)
        if (!capture_mode && has_viewport && m_editor_passes_enabled && scene_depth_decl) {
            if (m_grid_pass)
                m_grid_pass->render(*m_frame_graph, pass_ctx, scene_color_decl, scene_depth_decl);
            if (m_editor_pass) m_editor_pass->render(*m_frame_graph, pass_ctx);
        }
        if (!capture_mode) {
            if (m_lobe_pass) m_lobe_pass->render(*m_frame_graph, pass_ctx);
        }
    }

    // Declare reads on all debug target textures so frame graph tracks them.
    // Debug targets are created by the passes themselves -- we just look them up.
    std::vector<rendering::TextureDeclHandle> debug_target_decls;
    if (has_viewport) {
        auto collect_debug_targets = [&](auto& pass) {
            auto [targets, count] = pass.effective_debug_targets();
            for (uint32_t i = 0; i < count; ++i) {
                auto decl = m_frame_graph->find_texture(targets[i].resource_name);
                if (decl) {
                    debug_target_decls.push_back(decl);
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
        // ImGui overlay pass -- declare reads on any texture that ImGui::Image references
        auto imgui_builder = m_frame_graph->add_pass("imgui")
                                 .color(ctx.surface_view(), WGPUColor{0.08, 0.08, 0.12, 1.0})
                                 .present();
        if (has_viewport && display_color_decl) {
            imgui_builder.read(display_color_decl);
        }

        for (auto decl : debug_target_decls) {
            imgui_builder.read(decl);
        }

        // Declare read on gizmo overlay so ImGui can composite it
        if (has_viewport) {
            rendering::TextureDesc gizmo_desc;
            gizmo_desc.width = m_viewport_width;
            gizmo_desc.height = m_viewport_height;
            gizmo_desc.format = WGPUTextureFormat_RGBA8Unorm;
            gizmo_desc.clear_color = {0, 0, 0, 0};
            gizmo_overlay_decl = m_frame_graph->texture("editor_gizmo_overlay", gizmo_desc);
            if (gizmo_overlay_decl) {
                imgui_builder.read(gizmo_overlay_decl);
            }
        }

        {
            rendering::TextureDesc lobe_desc;
            lobe_desc.width = LobePass::k_texture_size;
            lobe_desc.height = LobePass::k_texture_size;
            lobe_desc.format = WGPUTextureFormat_RGBA8Unorm;
            lobe_desc.clear_color = {0.1, 0.1, 0.1, 1.0};
            auto lobe_color_decl = m_frame_graph->texture("lobe_color", lobe_desc);
            if (lobe_color_decl) {
                imgui_builder.read(lobe_color_decl);
            }
        }
        imgui_builder.execute([&](rendering::ExecuteContext&, WGPURenderPassEncoder pass) {
            m_imgui->end_frame(pass);
        });
    }

    m_frame_graph->compile();
    m_frame_graph->execute(ctx.encoder());

    // -- Issue capture readback (shared by --capture-and-quit and interactive screenshot) --
    {
        bool should_capture = false;
        if (capture_mode && m_frame_count >= m_app_config.capture_frames) {
            should_capture = true;
        } else if (m_screenshot_pending && has_viewport) {
            m_screenshot_pending = false;
            should_capture = true;
        }
        if (should_capture && !m_capture_readback.is_pending() && display_color_decl) {
            rendering::TextureDeclHandle target;
            if (m_debug_target_selection > 0 &&
                static_cast<size_t>(m_debug_target_selection - 1) < debug_target_decls.size()) {
                target = debug_target_decls[m_debug_target_selection - 1];
            } else {
                target = display_color_decl;
            }
            auto* target_compiled = m_frame_graph->compiled_texture(target);
            INVARIANT_MSG(target && target_compiled, "Capture target texture not available");
            m_capture_width = m_viewport_width;
            m_capture_height = m_viewport_height;
            m_capture_readback.request(ctx.encoder(), target_compiled->texture, m_capture_width,
                                       m_capture_height, device.handle(), device.instance());
        }
    }

    // -- GPU picking readback --
    m_picking_readback.tick();

    if (auto picked_id = m_picking_readback.try_read_u32()) {
        if (*picked_id == UINT32_MAX) {
            m_selected_prim = pxr::SdfPath();
        } else if (m_editor_pass) {
            const auto& path = m_editor_pass->resolve_picking_id(*picked_id);
            if (!path.IsEmpty()) {
                m_selected_prim = path;
                m_scroll_to_selected = true;
            }
        }
    }

    if (m_pick_requested && has_viewport && !m_picking_readback.is_pending()) {
        auto picking_decl = m_frame_graph->find_texture("picking_ids");
        auto* picking_compiled =
            picking_decl ? m_frame_graph->compiled_texture(picking_decl) : nullptr;

        if (picking_decl && picking_compiled && m_pick_x < m_viewport_width &&
            m_pick_y < m_viewport_height) {
            m_picking_readback.request(ctx.encoder(), picking_compiled->texture, m_pick_x, m_pick_y,
                                       device.handle(), device.instance());
            m_pick_requested = false;
        } else {
            m_pick_requested = false;
        }
    }

    if (!capture_mode) {
        // Store scene color view for next frame's ImGui::Image
        if (has_viewport && display_color_decl) {
            auto* dc = m_frame_graph->compiled_texture(display_color_decl);
            if (dc) m_scene_color_view = dc->view;
        }

        // Cache gizmo overlay view (must be after compile/execute)
        if (has_viewport && gizmo_overlay_decl) {
            auto* gc = m_frame_graph->compiled_texture(gizmo_overlay_decl);
            if (gc)
                m_gizmo_overlay_view = gc->view;
            else
                m_gizmo_overlay_view = nullptr;
        } else {
            m_gizmo_overlay_view = nullptr;
        }

        // Cache the active debug target view (selection 1 maps to debug_target_decls[0])
        if (m_debug_target_selection > 0 &&
            static_cast<size_t>(m_debug_target_selection - 1) < debug_target_decls.size()) {
            auto* dbg =
                m_frame_graph->compiled_texture(debug_target_decls[m_debug_target_selection - 1]);
            if (dbg)
                m_active_debug_view = dbg->view;
            else
                m_active_debug_view = nullptr;
        } else {
            m_active_debug_view = nullptr;
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
    ImGui::DockBuilderDockWindow("Renderer", down);
}

auto EditorApplication::create_input_actions() noexcept -> void {
    m_input_actions.clear();
}

auto EditorApplication::wrap_mouse_pos() noexcept -> void {
}

auto EditorApplication::draw_add_prim_menu(const pxr::SdfPath* parent,
                                           const glm::vec3* spawn_pos) noexcept -> void {
    PRECONDITION(m_stage);

    std::vector<rendering::PrimFactory> all_factories;
    for (auto* adapter : rendering::k_scene_adapters()) {
        auto factories = adapter->get_factories();
        all_factories.insert(all_factories.end(), factories.begin(), factories.end());
    }

    std::map<std::string, std::vector<const rendering::PrimFactory*>> grouped;
    for (const auto& f : all_factories) {
        grouped[f.category].push_back(&f);
    }

    for (const auto& [category, factories] : grouped) {
        if (ImGui::BeginMenu(category.c_str())) {
            for (const auto* factory : factories) {
                if (ImGui::MenuItem(factory->display_name.c_str())) {
                    auto path = find_unique_prim_path(factory->base_name, parent);
                    factory->define(m_stage, path);
                    normalize_xform_ops(path);
                    if (spawn_pos) {
                        auto prim = m_stage->GetPrimAtPath(path);
                        if (pxr::UsdGeomXformable xformable{prim}; xformable) {
                            bool reset = false;
                            auto ops = xformable.GetOrderedXformOps(&reset);
                            if (!ops.empty() &&
                                ops[0].GetOpType() == pxr::UsdGeomXformOp::TypeTransform) {
                                pxr::GfMatrix4d mat;
                                mat.SetIdentity();
                                mat.SetTranslateOnly(
                                    pxr::GfVec3d(spawn_pos->x, spawn_pos->y, spawn_pos->z));
                                ops[0].Set(mat);
                            }
                        }
                    }
                    m_selected_prim = path;
                }
            }
            ImGui::EndMenu();
        }
    }
}

void EditorApplication::load_stage(pxr::UsdStageRefPtr stage, std::string_view label) {
    INVARIANT_MSG(stage, "load_stage called with null stage");

    // Keep the built-in scene dropdown in sync with the actual loaded stage.
    // If the label matches a built-in entry we preselect it; otherwise -1 so
    // ImGui::Combo shows no entry as selected (and clicking any entry becomes
    // a real switch, not a silent no-op).
    m_demo_scene_index = -1;
    for (size_t i = 0; i < m_demo_scene_names.size(); ++i) {
        if (m_demo_scene_names[i] == label) {
            m_demo_scene_index = static_cast<int>(i);
            break;
        }
    }

    if (m_init_complete) {
        // Async path -- populate in background, finalize in render()
        m_scene_load_task.reset();
        m_pending_stage.Reset();
        m_pending_stage = stage;

        m_scene_load_task = std::make_unique<OneShotTask<rendering::RenderWorld>>(
            "Loading Scene", [stage](TaskProgress& progress) -> rendering::RenderWorld {
                return rendering::populate_from_stage(stage, progress);
            });

        m_loading_overlay.track({
            "Loading Scene",
            [this] { return !m_scene_load_task || m_scene_load_task->is_done(); },
            [this] { return m_scene_load_task ? m_scene_load_task->progress() : 1.0f; },
            [this] { return m_scene_load_task ? m_scene_load_task->status() : std::string{}; },
        });

        log(LogLevel::Info, "Loading scene: {} (background)", label);
    } else {
        // Sync path -- during on_ready(), before init is complete
        auto const& device = webgpu_context()->device();

        // Apply override layer if specified (only on initial load)
        if (!m_app_config.usd_override_path.empty()) {
            auto session = stage->GetSessionLayer();
            session->InsertSubLayerPath(m_app_config.usd_override_path);
            log(LogLevel::Info, "Applied USD override: {}", m_app_config.usd_override_path);
        }

        rendering::populate_from_stage(m_world, stage);
        m_world.upload_all_meshes(device);
        m_stage = stage;
        activate_stage();
        log(LogLevel::Info, "Loaded scene: {} ({} objects)", label, m_world.get_objects().size());
    }
}

void EditorApplication::activate_stage() {
    PRECONDITION(m_stage);

    // Read stage metadata
    m_stage_settings = rendering::read_stage_settings(m_stage);

    // Update camera for stage coordinate system
    m_camera.set_up_axis(m_stage_settings.up_axis);
    m_camera.apply_meters_per_unit(m_stage_settings.meters_per_unit);

    // Stage listener
    revoke_stage_listener();
    register_stage_listener();

    ensure_default_light();

    // Reset selection/picking state
    m_selected_prim = pxr::SdfPath();
    m_picking_readback = webgpu::BufferReadback{};
    m_pick_requested = false;
    m_first_prep = true;
}

auto EditorApplication::draw_scene_panel() noexcept -> void {
    ImGui::TextUnformatted(k_editor_tutorial_text);
    ImGui::Separator();

    if (!m_demo_scene_paths.empty()) {
        ImGui::SetNextItemWidth(160.0f);
        auto count = static_cast<int>(m_demo_scene_paths.size());
        auto* names = &m_demo_scene_names;
        if (ImGui::Combo(
                "##demo_scene", &m_demo_scene_index,
                [](void* data, int idx) -> const char* {
                    return (*static_cast<std::vector<std::string>*>(data))[idx].c_str();
                },
                names, count)) {
            auto stage = pxr::UsdStage::Open(m_demo_scene_paths[m_demo_scene_index]);
            INVARIANT_MSG(stage, "Failed to open demo scene");
            load_stage(stage, m_demo_scene_names[m_demo_scene_index]);
        }
        ImGui::SameLine();
    }

    if (ImGui::Button("Open Scene")) open_scene_dialog();

    if (m_stage) {
        ImGui::SameLine();
        if (ImGui::Button("Save Scene")) save_scene_dialog();
    }
}

void EditorApplication::open_scene_dialog() {
    ImGui::FileDialogueAsync(
        ImGui::FileDialogueMode::Open, ".usdz,.usda,.usdc,.usd",
        [this](ImGui::FileDialogueResult result) {
            pxr::UsdStageRefPtr stage;
#ifdef __EMSCRIPTEN__
            auto memfs_path = "/tmp/" + result.name;
            {
                std::ofstream ofs(memfs_path, std::ios::binary);
                CHECK_MSG(ofs.is_open(), "Failed to open MEMFS path for writing");
                ofs.write(result.contents.data(),
                          static_cast<std::streamsize>(result.contents.size()));
                CHECK_MSG(ofs.good(), "Failed to write uploaded file to MEMFS");
            }
            if (!m_memfs_path.empty()) std::remove(m_memfs_path.c_str());
            m_memfs_path = memfs_path;
            stage = pxr::UsdStage::Open(memfs_path);
#else
            stage = pxr::UsdStage::Open(result.name);
#endif
            if (!stage) {
                log(LogLevel::Error, "Failed to open stage: {}", result.name);
                return;
            }
            load_stage(stage, result.name);
        });
}

void EditorApplication::save_scene_dialog() {
#ifdef __EMSCRIPTEN__
    std::string const out_path = "/tmp/_pts_export.usdz";
    bool const saved = pts::rendering::save_stage(m_stage, out_path);
    CHECK_MSG(saved, "Failed to save stage for USDZ download");

    std::FILE* f = std::fopen(out_path.c_str(), "rb");
    CHECK_MSG(f, "Failed to open USDZ temp file for reading");
    std::fseek(f, 0, SEEK_END);
    auto const nbytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> bytes(nbytes);
    std::fread(bytes.data(), 1, nbytes, f);
    std::fclose(f);

    // clang-format off
    EM_ASM({
        var data = HEAPU8.subarray($0, $0 + $1);
        var blob = new Blob([new Uint8Array(data)], {type: 'application/octet-stream'});
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'scene.usdz';
        a.click();
        URL.revokeObjectURL(a.href);
    }, bytes.data(), bytes.size());
    // clang-format on

    std::remove(out_path.c_str());
    log(LogLevel::Info, "USDZ download triggered");
#else
    ImGui::FileDialogueAsync(ImGui::FileDialogueMode::Save, ".usdz,.usda,.usdc,.usd",
                             [this](ImGui::FileDialogueResult result) {
                                 if (result.name.empty()) return;
                                 if (pts::rendering::save_stage(m_stage, result.name)) {
                                     log(LogLevel::Info, "Saved scene to {}", result.name);
                                 } else {
                                     log(LogLevel::Error, "Failed to save scene to {}",
                                         result.name);
                                 }
                             });
#endif
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

    if (m_stage &&
        ImGui::BeginPopupContextWindow("SceneContextMenu", ImGuiPopupFlags_NoOpenOverItems |
                                                               ImGuiPopupFlags_MouseButtonRight)) {
        draw_add_prim_menu();
        ImGui::EndPopup();
    }

    ImGui::Separator();
    if (!m_selected_prim.IsEmpty()) {
        auto prim = m_stage->GetPrimAtPath(m_selected_prim);
        if (prim.IsValid()) {
            // -- Transform section (TRS) --
            pxr::UsdGeomXformable xformable(prim);
            if (xformable) {
                pxr::GfMatrix4d gf_local;
                bool resetsXformStack;
                xformable.GetLocalTransformation(&gf_local, &resetsXformStack,
                                                 pxr::UsdTimeCode::Default());

                glm::mat4 local_mat;
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        local_mat[i][j] = static_cast<float>(gf_local[i][j]);

                auto trs = decompose_trs(local_mat);

                if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
                    bool changed = false;
                    changed |= ImGui::DragFloat3("Translate", &trs.translate.x, 0.01f);
                    changed |= ImGui::DragFloat3("Rotate", &trs.rotate_degrees.x, 0.5f);
                    changed |= ImGui::DragFloat3("Scale", &trs.scale.x, 0.01f);

                    if (changed) {
                        normalize_xform_ops(m_selected_prim);
                        bool reset = false;
                        auto ops = xformable.GetOrderedXformOps(&reset);
                        INVARIANT(ops.size() == 1);
                        glm::mat4 new_local = compose_trs(trs);
                        pxr::GfMatrix4d new_gf;
                        for (int i = 0; i < 4; ++i)
                            for (int j = 0; j < 4; ++j)
                                new_gf[i][j] = static_cast<double>(new_local[i][j]);
                        ops[0].Set(new_gf);
                    }
                }
                ImGui::Spacing();
            }

            draw_prim_properties(prim);

            // Show BRDF lobe viewer if prim has a bound material
            auto bound_mat = pxr::UsdShadeMaterialBindingAPI(prim).ComputeBoundMaterial();
            if (bound_mat && m_lobe_pass) {
                auto surface = bound_mat.GetSurfaceOutput();
                pxr::UsdShadeConnectableAPI source;
                pxr::TfToken source_name;
                pxr::UsdShadeAttributeType source_type;
                if (surface.GetConnectedSource(&source, &source_name, &source_type)) {
                    auto shader = pxr::UsdShadeShader(source.GetPrim());

                    // Sync from USD -> lobe sliders only when selection changes
                    if (m_selected_prim != m_lobe_bound_prim) {
                        m_lobe_bound_prim = m_selected_prim;
                        float roughness = 0.5f;
                        float metallic = 0.0f;
                        float val;
                        if (auto r = shader.GetInput(pxr::TfToken("roughness")); r && r.Get(&val))
                            roughness = val;
                        if (auto m = shader.GetInput(pxr::TfToken("metallic")); m && m.Get(&val))
                            metallic = val;
                        m_lobe_pass->set_material(roughness, metallic);
                    }

                    ImGui::Separator();
                    if (ImGui::CollapsingHeader("BRDF Lobe", ImGuiTreeNodeFlags_DefaultOpen)) {
                        if (m_lobe_pass->draw_lobe_widget()) {
                            // Write changed values back to USD
                            if (auto r = shader.GetInput(pxr::TfToken("roughness")); r)
                                r.Set(m_lobe_pass->roughness());
                            if (auto m = shader.GetInput(pxr::TfToken("metallic")); m)
                                m.Set(m_lobe_pass->metallic());
                        }
                    }
                }
            }
        }
    } else {
        m_lobe_bound_prim = pxr::SdfPath();
        ImGui::TextDisabled("Select a prim to inspect properties");
    }
}

void EditorApplication::draw_prim_tree(const pxr::UsdPrim& prim) {
    auto path = prim.GetPath();
    auto name = prim.GetName().GetString();
    auto type_name = prim.GetTypeName().GetString();

    bool is_selected = (m_selected_prim == path);
    bool is_renaming = (m_renaming_prim == path);
    bool has_children = !prim.GetChildren().empty();

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;
    if (!has_children) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

    // Auto-open parent nodes when a child is selected (e.g. via picking)
    if (!is_selected && !m_selected_prim.IsEmpty() && m_selected_prim.HasPrefix(path)) {
        ImGui::SetNextItemOpen(true);
    }

    bool node_open;
    if (is_renaming) {
        // Render tree node with blank label, overlay InputText for rename
        node_open = ImGui::TreeNodeEx(path.GetText(), flags, " ");

        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        if (!m_rename_focus_set) {
            ImGui::SetKeyboardFocusHere();
            m_rename_focus_set = true;
        }

        bool apply = ImGui::InputText(
            "##rename", m_rename_buf, sizeof(m_rename_buf),
            ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
        bool cancel = ImGui::IsKeyPressed(ImGuiKey_Escape);
        bool lost_focus =
            !ImGui::IsItemActive() && m_rename_focus_set && ImGui::GetFrameCount() > 1;

        if (apply) {
            std::string new_name(m_rename_buf);
            if (!new_name.empty() && new_name != name) {
                auto layer = m_stage->GetRootLayer();
                auto spec = layer->GetPrimAtPath(m_renaming_prim);
                CHECK_MSG(spec, "prim being renamed must have a spec in the root layer");
                if (spec->SetName(new_name)) {
                    auto parent = m_renaming_prim.GetParentPath();
                    m_selected_prim = parent.AppendChild(pxr::TfToken(new_name));
                }
            }
            m_renaming_prim = pxr::SdfPath();
        } else if (cancel || lost_focus) {
            m_renaming_prim = pxr::SdfPath();
        }
    } else {
        // Normal tree node rendering
        std::string label = type_name.empty() ? name : name + " (" + type_name + ")";
        node_open = ImGui::TreeNodeEx(path.GetText(), flags, "%s", label.c_str());

        // Auto-scroll to the selected prim once (when selection changes via picking)
        if (is_selected && m_scroll_to_selected) {
            ImGui::ScrollToItem();
            m_scroll_to_selected = false;
        }

        // Double-click to rename
        bool double_clicked = ImGui::IsItemHovered() &&
                              ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) &&
                              !ImGui::IsItemToggledOpen();

        if (double_clicked) {
            m_renaming_prim = path;
            m_selected_prim = path;
            std::strncpy(m_rename_buf, name.c_str(), sizeof(m_rename_buf) - 1);
            m_rename_buf[sizeof(m_rename_buf) - 1] = '\0';
            m_rename_focus_set = false;
        } else if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
            if (is_selected) {
                m_selected_prim = pxr::SdfPath();
            } else {
                m_selected_prim = path;
            }
        }

        // F2 to rename selected prim
        if (is_selected && ImGui::IsKeyPressed(ImGuiKey_F2) && !ImGui::GetIO().WantTextInput) {
            m_renaming_prim = path;
            std::strncpy(m_rename_buf, name.c_str(), sizeof(m_rename_buf) - 1);
            m_rename_buf[sizeof(m_rename_buf) - 1] = '\0';
            m_rename_focus_set = false;
        }

        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::BeginMenu("Add Child")) {
                draw_add_prim_menu(&path);
                ImGui::EndMenu();
            }
            if (ImGui::MenuItem("Delete")) {
                m_stage->RemovePrim(path);
                if (m_selected_prim == path || m_selected_prim.HasPrefix(path)) {
                    m_selected_prim = pxr::SdfPath();
                }
            }
            if (ImGui::MenuItem("Rename", "F2")) {
                m_renaming_prim = path;
                m_selected_prim = path;
                std::strncpy(m_rename_buf, name.c_str(), sizeof(m_rename_buf) - 1);
                m_rename_buf[sizeof(m_rename_buf) - 1] = '\0';
                m_rename_focus_set = false;
            }
            ImGui::EndPopup();
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
    m_viewport_combo_open = false;
    if (ImGui::BeginMenuBar()) {
        ImGui::Text("Renderer:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        auto& reg_entries = rendering::RendererRegistry::entries();
        if (ImGui::BeginCombo("##renderer", reg_entries[m_active_config_index].name.c_str())) {
            m_viewport_combo_open = true;
            for (size_t i = 0; i < reg_entries.size(); ++i) {
                bool selected = (i == m_active_config_index);
                if (ImGui::Selectable(reg_entries[i].name.c_str(), selected)) {
                    if (i != m_active_config_index) {
                        create_renderer(i);
                    }
                }
            }
            ImGui::EndCombo();
        }
        // Exposure control
        if (m_renderer_pass) {
            ImGui::SameLine();
            ImGui::Text("EV:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(80);
            ImGui::DragFloat("##exposure", &m_renderer_pass->exposure(), 0.05f, -5.0f, 5.0f,
                             "%.1f");
        }
        // Renderer-specific controls
        if (m_renderer_pass) {
            m_renderer_pass->draw_viewport_controls();
        }
        ImGui::SameLine();
        ImGui::Checkbox("Gizmos", &m_editor_passes_enabled);
        // "..." overflow menu
        ImGui::SameLine();
        if (ImGui::SmallButton("...")) {
            ImGui::OpenPopup("##toolbar_more");
        }
        if (ImGui::BeginPopup("##toolbar_more")) {
            // Debug Output submenu
            if (ImGui::BeginMenu("Debug Output")) {
                std::vector<std::string> debug_labels;
                debug_labels.emplace_back("Off");
                for_each_pass([&](auto& pass) {
                    auto [targets, count] = pass.effective_debug_targets();
                    for (uint32_t i = 0; i < count; ++i) {
                        debug_labels.emplace_back(std::string(pass.name()) + ": " +
                                                  targets[i].label);
                    }
                });
                if (m_debug_target_selection >= static_cast<int>(debug_labels.size())) {
                    m_debug_target_selection = 0;
                }
                for (int i = 0; i < static_cast<int>(debug_labels.size()); ++i) {
                    bool selected = (i == m_debug_target_selection);
                    if (ImGui::MenuItem(debug_labels[i].c_str(), nullptr, selected)) {
                        m_debug_target_selection = i;
                    }
                }
                ImGui::EndMenu();
            }
            // Camera submenu
            if (ImGui::BeginMenu("Camera")) {
                const auto& cameras = m_world.get_cameras();
                std::vector<std::pair<std::string, int>> cam_labels;
                cam_labels.push_back({"Free Camera", 0});
                cameras.for_each([&](const pxr::SdfPath& path, const rendering::CameraData&) {
                    auto idx = cameras.find(path).index();
                    cam_labels.push_back({path.GetName(), static_cast<int>(idx + 1)});
                });
                for (auto& [label, idx] : cam_labels) {
                    bool selected = (idx == m_active_camera_index);
                    if (ImGui::MenuItem(label.c_str(), nullptr, selected)) {
                        m_active_camera_index = idx;
                    }
                }
                ImGui::EndMenu();
            }
            // Save View (only when free camera is active)
            if (m_stage && m_active_camera_index == 0) {
                if (ImGui::MenuItem("Save View")) {
                    auto cam_path = find_unique_prim_path("Camera");
                    rendering::CameraAdapter::create_from_view(
                        m_stage, cam_path, m_camera.view_matrix(),
                        glm::radians(m_camera.fov_y_degrees()), m_camera.near_plane(),
                        m_camera.far_plane());
                }
            }
            // Capture
            if (ImGui::MenuItem("Capture")) {
                m_screenshot_pending = true;
            }
            ImGui::EndPopup();
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
        auto display_view = (m_debug_target_selection > 0 && m_active_debug_view)
                                ? m_active_debug_view
                                : m_scene_color_view;
        if (display_view && m_viewport_width > 0 && m_viewport_height > 0) {
            ImGui::PushID("viewport_image");
            ImGui::Image(reinterpret_cast<ImTextureID>(display_view),
                         ImVec2(static_cast<float>(m_viewport_width),
                                static_cast<float>(m_viewport_height)));
            ImGui::PopID();
            // Overlay gizmo wireframes on top (visible in all views including debug)
            if (m_gizmo_overlay_view && m_editor_passes_enabled) {
                auto* draw_list = ImGui::GetWindowDrawList();
                ImVec2 p_min(m_viewport_x, m_viewport_y);
                ImVec2 p_max(m_viewport_x + static_cast<float>(m_viewport_width),
                             m_viewport_y + static_cast<float>(m_viewport_height));
                draw_list->AddImage(reinterpret_cast<ImTextureID>(m_gizmo_overlay_view), p_min,
                                    p_max);
            }
            // Draw renderer debug overlays (e.g. BVH wireframes)
            if (m_renderer_pass && m_viewport_width > 0 && m_viewport_height > 0) {
                auto view = compute_active_view(static_cast<float>(m_viewport_width) /
                                                static_cast<float>(m_viewport_height));
                rendering::IPass::ViewportOverlayParams overlay_params{
                    view.proj_matrix * view.view_matrix, m_viewport_x, m_viewport_y,
                    static_cast<float>(m_viewport_width), static_cast<float>(m_viewport_height)};
                m_renderer_pass->draw_viewport_overlay(overlay_params);
            }
        } else {
            ImGui::TextUnformatted("Renderer output not available");
        }
    }

    // -- ImGuizmo gizmo --
    if (m_editor_passes_enabled && !m_selected_prim.IsEmpty() && m_stage && m_viewport_width > 0 &&
        m_viewport_height > 0) {
        auto prim = m_stage->GetPrimAtPath(m_selected_prim);
        pxr::UsdGeomXformable xformable(prim);
        if (prim.IsValid() && xformable) {
            float aspect =
                static_cast<float>(m_viewport_width) / static_cast<float>(m_viewport_height);
            auto [view_mat, proj_mat, cam_pos] = compute_active_view(aspect);

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
                if (m_selected_prim != m_xform_normalized_prim) {
                    normalize_xform_ops(m_selected_prim);
                    m_xform_normalized_prim = m_selected_prim;
                }

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

    // -- Viewport right-click context menu --
    if (m_open_viewport_context) {
        m_open_viewport_context = false;
        ImGui::OpenPopup("ViewportContextMenu");
    }
    if (m_stage && ImGui::BeginPopup("ViewportContextMenu")) {
        draw_add_prim_menu(nullptr, &m_context_menu_world_pos);
        ImGui::EndPopup();
    }
}

auto EditorApplication::draw_console_panel() noexcept -> void {
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
        if (msgs.size() != m_last_console_msg_count) {
            ImGui::SetScrollHereY(1.0f);
            m_last_console_msg_count = msgs.size();
        }
    }
    ImGui::EndChild();
}

auto EditorApplication::on_mouse_leave_scene_viewport() noexcept -> void {
}

auto EditorApplication::on_mouse_enter_scene_viewport() noexcept -> void {
}

auto EditorApplication::handle_input(InputEvent const& event) noexcept -> void {
    // Picking works regardless of camera mode
    if (event.input.input_type == InputType::MOUSE &&
        event.input.key_or_button == ImGuiMouseButton_Left &&
        event.input.action_type == ActionType::PRESS && !ImGuizmo::IsOver() &&
        !m_viewport_combo_open) {
        auto local_x = event.mouse_pos.x - m_viewport_x;
        auto local_y = event.mouse_pos.y - m_viewport_y;
        if (local_x >= 0 && local_y >= 0 && local_x < static_cast<float>(m_viewport_width) &&
            local_y < static_cast<float>(m_viewport_height)) {
            m_pick_x = static_cast<uint32_t>(local_x);
            m_pick_y = static_cast<uint32_t>(local_y);
            m_pick_requested = true;
        }
    }

    if (m_active_camera_index != 0) return;  // scene camera active -- no orbit input

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

    // Delete key: remove selected prim (works from any window, guarded against text input)
    if (event.input.input_type == InputType::KEYBOARD &&
        event.input.action_type == ActionType::PRESS &&
        event.input.key_or_button == ImGuiKey_Delete && !m_selected_prim.IsEmpty() && m_stage &&
        !ImGui::GetIO().WantTextInput) {
        m_stage->RemovePrim(m_selected_prim);
        m_selected_prim = pxr::SdfPath();
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
        // Right-click: orbit camera on drag, context menu on click
        if (event.input.key_or_button == ImGuiMouseButton_Right) {
            if (event.input.action_type == ActionType::PRESS) {
                m_rmb_dragged = false;
                m_rmb_press_pos = event.mouse_pos;
            } else if (event.input.action_type == ActionType::HOLD) {
                constexpr float k_drag_threshold = 3.0f;
                if (glm::length(event.mouse_pos - m_rmb_press_pos) > k_drag_threshold) {
                    m_rmb_dragged = true;
                }
                if (m_rmb_dragged) {
                    m_camera.orbit(event.normalized_mouse_delta.x, event.normalized_mouse_delta.y);
                }
            } else if (event.input.action_type == ActionType::RELEASE && !m_rmb_dragged) {
                // Right-click without drag: open viewport context menu
                // Intersect click ray with the ground plane (up_axis = 0)
                float aspect = static_cast<float>(m_viewport_width) /
                               std::max(1.0f, static_cast<float>(m_viewport_height));
                auto local_x = m_rmb_press_pos.x - m_viewport_x;
                auto local_y = m_rmb_press_pos.y - m_viewport_y;
                float ndc_x = (local_x / static_cast<float>(m_viewport_width)) * 2.0f - 1.0f;
                float ndc_y = 1.0f - (local_y / static_cast<float>(m_viewport_height)) * 2.0f;
                auto inv_vp =
                    glm::inverse(m_camera.projection_matrix(aspect) * m_camera.view_matrix());
                glm::vec4 near_h = inv_vp * glm::vec4(ndc_x, ndc_y, 0.0f, 1.0f);
                glm::vec4 far_h = inv_vp * glm::vec4(ndc_x, ndc_y, 1.0f, 1.0f);
                glm::vec3 near_pt = glm::vec3(near_h) / near_h.w;
                glm::vec3 far_pt = glm::vec3(far_h) / far_h.w;
                glm::vec3 ray_dir = far_pt - near_pt;
                // Y-up: intersect with y=0 plane; fall back to fixed distance
                float t = (std::abs(ray_dir.y) > 1e-6f) ? (-near_pt.y / ray_dir.y) : -1.0f;
                if (t < 0.0f) t = m_camera.distance() / glm::length(ray_dir);
                m_context_menu_world_pos = near_pt + ray_dir * t;
                m_open_viewport_context = true;
            }
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
