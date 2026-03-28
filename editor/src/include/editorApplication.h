#pragma once

#include <core/gpuApplication.h>
#include <core/imgui/loadingOverlay.h>
#include <core/inputAction.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/preparedSceneData.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/webgpu/bufferReadback.h>
#include <core/rendering/webgpu/textureReadback.h>
#include <core/rendering/webgpu/webgpu.h>
#include <core/worker.h>
#include <pxr/base/tf/notice.h>
#include <pxr/base/tf/weakBase.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/notice.h>
#include <pxr/usd/usd/stage.h>
#include <spdlog/sinks/ringbuffer_sink.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "perfOverlay.h"

namespace pts {
class ImGuiComponent;
class InputComponent;
}  // namespace pts

namespace pts::rendering {
class IRenderPass;
class IRenderer;
}  // namespace pts::rendering
namespace pts::editor {
class EditorPass;
class LobePass;
class ToneMappingPass;
}  // namespace pts::editor

namespace pts::editor {

struct AppConfig {
    std::string capture_output;     // empty = no capture mode
    std::string usd_path;           // empty = embedded default
    std::string usd_override_path;  // empty = no override layer
    int capture_frames = 1;         // frames to render before capture
    std::string renderer_name;      // empty = default (first)
    std::string debug_output_name;  // empty = scene_color
    std::string camera_target;      // "x,y,z" — empty = default
    std::string camera_distance;    // empty = default (3.0)
    std::string camera_yaw;         // degrees, empty = default (0)
    std::string camera_pitch;       // degrees, empty = default (~17)
    std::string camera_fov;         // degrees, empty = default (60)
    std::string camera_prim_path;   // scene camera prim name, empty = free camera

    [[nodiscard]] bool is_capture_mode() const {
        return !capture_output.empty();
    }
};

struct EditorApplication final : GpuApplication {
    NO_COPY_MOVE(EditorApplication);

    EditorApplication(std::string_view name, pts::LoggingManager& logging_manager);
    ~EditorApplication() override;

    void register_args(CommandLine& cli) override;
    void process_args(const CommandLine& cli) override;

   protected:
    void on_ready() override;
    void update(float dt) override;
    void render(FrameContext& ctx) override;

   private:
    struct ActiveView {
        glm::mat4 view_matrix;
        glm::mat4 proj_matrix;
        glm::vec3 camera_position;
    };
    ActiveView compute_active_view(float aspect) const;

    void setup_docking_layout();
    void create_renderer(size_t index);
    auto create_input_actions() noexcept -> void;
    auto wrap_mouse_pos() noexcept -> void;

    // Scene I/O
    void load_stage(pxr::UsdStageRefPtr stage, std::string_view label);
    void activate_stage();
    void open_scene_dialog();
    void save_scene_dialog();

    // Capture
    void save_capture_png(boost::span<const uint8_t> pixels, std::string_view path);

    // imgui rendering
    auto draw_scene_panel() noexcept -> void;
    auto draw_inspector_panel() noexcept -> void;
    void draw_prim_tree(const pxr::UsdPrim& prim);
    auto draw_scene_viewport() noexcept -> void;
    auto draw_console_panel() noexcept -> void;
    // events
    auto on_mouse_leave_scene_viewport() noexcept -> void;
    auto on_mouse_enter_scene_viewport() noexcept -> void;

    auto handle_input(InputEvent const& event) noexcept -> void;

    // Components
    std::unique_ptr<ImGuiComponent> m_imgui;
    std::unique_ptr<InputComponent> m_input;

    AppConfig m_app_config;
    rendering::StageSettings m_stage_settings;
    bool m_init_complete{false};

    std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> m_console_log_sink;

    // input handling
    std::vector<InputAction> m_input_actions;

    bool m_first_frame{true};

    // Rendering
    std::unique_ptr<rendering::FrameGraph> m_frame_graph;
    rendering::OrbitCamera m_camera;
    int m_active_camera_index = 0;  // 0 = free camera, 1..N = scene cameras
    rendering::RenderWorld m_world;

    // Async CPU scene preparation
    struct CpuPrepJob {};
    std::unique_ptr<Worker<CpuPrepJob, rendering::PreparedSceneData>> m_prep_worker;
    bool m_first_prep{true};

    std::unique_ptr<rendering::IRenderer> m_renderer_pass;
    std::vector<std::unique_ptr<rendering::IRenderPass>> m_editor_passes;
    EditorPass* m_editor_pass = nullptr;  // non-owning, points into m_editor_passes
    LobePass* m_lobe_pass = nullptr;      // non-owning, points into m_editor_passes
    rendering::IRenderPass* m_tonemapping_pass =
        nullptr;  // non-owning, points into m_editor_passes
    size_t m_active_config_index = 0;
    bool m_editor_passes_enabled = true;
    rendering::ShaderLoader m_shader_loader;

    /// Iterate all active passes (renderer + editor) in execution order.
    template <typename Fn>
    void for_each_pass(Fn&& fn) {
        if (m_renderer_pass) fn(*m_renderer_pass);
        for (auto& p : m_editor_passes) {
            // ToneMappingPass always runs; others respect the toggle
            if (!m_editor_passes_enabled && p.get() != m_tonemapping_pass) continue;
            fn(*p);
        }
    }

    // USD stage + change tracking
    pxr::UsdStageRefPtr m_stage;

    struct StageListener : pxr::TfWeakBase {
        using Callback = void (*)(void*, const pxr::UsdNotice::ObjectsChanged&);
        void* ctx{};
        Callback cb{};
        void handle(const pxr::UsdNotice::ObjectsChanged& notice,
                    const pxr::UsdStageWeakPtr& sender);
    };

    StageListener m_stage_listener;
    pxr::TfNotice::Key m_listener_key;

    void register_stage_listener();
    void revoke_stage_listener();
    void on_objects_changed(const pxr::UsdNotice::ObjectsChanged& notice);
    void process_dirty_prims();
    void normalize_xform_ops(const pxr::SdfPath& prim_path);
    pxr::SdfPath find_unique_prim_path(std::string_view base_name,
                                       const pxr::SdfPath* parent = nullptr);
    auto draw_add_prim_menu(const pxr::SdfPath* parent = nullptr,
                            const glm::vec3* spawn_pos = nullptr) noexcept -> void;
    void ensure_default_light();

    std::vector<pxr::SdfPath> m_resync_paths;
    std::vector<pxr::SdfPath> m_dirty_xform_paths;

    // Selection & gizmo
    pxr::SdfPath m_selected_prim;
    pxr::SdfPath m_xform_normalized_prim;  // last prim whose xform was normalized for gizmo
    pxr::SdfPath m_lobe_bound_prim;        // tracks which prim's material is loaded in lobe viewer
    enum class GizmoOp { Translate, Rotate, Scale };
    GizmoOp m_gizmo_op = GizmoOp::Translate;

    // Inline rename state
    pxr::SdfPath m_renaming_prim;
    char m_rename_buf[256]{};
    bool m_rename_focus_set = false;

    // Viewport tracking
    uint32_t m_viewport_width = 0;
    uint32_t m_viewport_height = 0;
    float m_viewport_x = 0.0f;
    float m_viewport_y = 0.0f;
    rendering::TextureRef m_scene_color_ref;

    // Debug visualization
    bool m_viewport_combo_open =
        false;  // suppresses picking while combo dropdown overlaps viewport
    int m_debug_target_selection = 0;
    rendering::TextureRef m_active_debug_ref;
    rendering::TextureRef m_gizmo_overlay_ref;

    // Console auto-scroll
    size_t m_last_console_msg_count = 0;

    // Viewport context menu
    bool m_rmb_dragged = false;            // true if right-click has moved beyond threshold
    bool m_open_viewport_context = false;  // deferred popup open (set in input, read in draw)
    glm::vec2 m_rmb_press_pos{0, 0};       // screen pos at right-click press
    glm::vec3 m_context_menu_world_pos{0, 0, 0};  // 3D spawn point for context menu

    // GPU picking
    webgpu::BufferReadback m_picking_readback;
    bool m_pick_requested = false;
    uint32_t m_pick_x = 0;
    uint32_t m_pick_y = 0;

    // Performance overlay
    PerfOverlay m_perf_overlay;

    // Capture mode state
    int m_frame_count = 0;
    webgpu::TextureReadback m_capture_readback;
    bool m_screenshot_pending{false};

    // Async scene loading
    std::unique_ptr<pts::OneShotTask<rendering::RenderWorld>> m_scene_load_task;
    pxr::UsdStageRefPtr m_pending_stage;

    std::vector<std::string> m_demo_scene_paths;
    std::vector<std::string> m_demo_scene_names;
    int m_demo_scene_index = 0;
#ifdef __EMSCRIPTEN__
    std::string m_memfs_path;  // last uploaded file in MEMFS, for cleanup
#endif

    // Loading overlay
    pts::LoadingOverlay m_loading_overlay;
};
}  // namespace pts::editor
