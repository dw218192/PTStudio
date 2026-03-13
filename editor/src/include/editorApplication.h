#pragma once

#include <core/inputAction.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/webgpu/webgpu.h>
#include <core/windowedApplication.h>
#include <pxr/base/tf/notice.h>
#include <pxr/base/tf/weakBase.h>
#include <pxr/usd/usd/notice.h>
#include <pxr/usd/usd/stage.h>
#include <spdlog/sinks/ringbuffer_sink.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pts {
class ImGuiComponent;
class InputComponent;
}  // namespace pts

namespace pts::rendering {
class IScenePass;
}

namespace pts::editor {
struct AppConfig {
    bool quit_on_start{false};
};

struct EditorApplication final : WindowedApplication {
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
    void setup_docking_layout();
    void set_renderer_config(size_t index);
    auto create_input_actions() noexcept -> void;
    auto wrap_mouse_pos() noexcept -> void;

    // imgui rendering
    auto draw_scene_panel() noexcept -> void;
    auto draw_object_panel() noexcept -> void;
    auto draw_scene_viewport() noexcept -> void;
    auto draw_console_panel() const noexcept -> void;

    // events
    auto on_mouse_leave_scene_viewport() noexcept -> void;
    auto on_mouse_enter_scene_viewport() noexcept -> void;

    auto handle_input(InputEvent const& event) noexcept -> void;

    // Components
    std::unique_ptr<ImGuiComponent> m_imgui;
    std::unique_ptr<InputComponent> m_input;

    AppConfig m_app_config;

    std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> m_console_log_sink;

    // input handling
    std::vector<InputAction> m_input_actions;

    bool m_first_frame{true};

    // Rendering
    std::unique_ptr<rendering::FrameGraph> m_frame_graph;
    rendering::OrbitCamera m_camera;
    rendering::RenderWorld m_world;
    std::vector<std::unique_ptr<rendering::IScenePass>> m_passes;
    size_t m_active_config_index = 0;

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

    bool m_needs_full_resync{false};
    std::vector<std::string> m_dirty_xform_paths;

    // Selection & gizmo
    int m_selected_object = -1;
    enum class GizmoOp { Translate, Rotate, Scale };
    GizmoOp m_gizmo_op = GizmoOp::Translate;

    // Viewport tracking
    uint32_t m_viewport_width = 0;
    uint32_t m_viewport_height = 0;
    float m_viewport_x = 0.0f;
    float m_viewport_y = 0.0f;
    rendering::TextureRef m_scene_color_ref;
};
}  // namespace pts::editor
