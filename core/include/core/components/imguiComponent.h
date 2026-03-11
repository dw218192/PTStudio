#pragma once

#include <core/defines.h>
#include <core/rendering/webgpu/webgpu.h>
#include <core/signal.h>
#include <imgui.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace pts {
class LoggingManager;

namespace rendering {
class IViewport;
class WebGpuContext;
class IImguiWindowing;
class IImguiRendering;
}  // namespace rendering

class ImGuiComponent {
   public:
    NO_COPY_MOVE(ImGuiComponent);

    struct WindowInfo {
        Signal<void()> on_leave_region;
        Signal<void()> on_enter_region;
    };

    /// RAII scope that pairs begin_frame / end_frame.
    /// Destructor calls end_frame() (full render) unless render_into() was called first.
    class FrameScope {
        NO_COPY_MOVE(FrameScope);

       public:
        ~FrameScope();

        /// Finalize ImGui and render draw data into an existing render pass.
        /// After this call, the destructor is a no-op.
        void render_into(WGPURenderPassEncoder pass);

       private:
        friend class ImGuiComponent;
        explicit FrameScope(ImGuiComponent& owner);
        ImGuiComponent* m_owner;
    };

    explicit ImGuiComponent(rendering::IViewport& viewport,
                            rendering::WebGpuContext& webgpu_context,
                            LoggingManager& logging_manager);
    ~ImGuiComponent();

    /// Begin a new ImGui frame and return an RAII scope guard.
    [[nodiscard]] FrameScope frame_scope();

    void begin_frame();

    /// Finalize and render using own surface + command encoder (editor pattern).
    void end_frame();

    /// Finalize and render into an existing render pass (overlay pattern).
    void end_frame(WGPURenderPassEncoder pass);

    [[nodiscard]] bool is_ready() const noexcept;

    auto get_window_info(std::string_view name) noexcept -> WindowInfo&;
    auto begin_window(std::string_view name, ImGuiWindowFlags flags = 0) noexcept -> bool;
    void end_window() noexcept;
    auto get_window_content_pos(std::string_view name) const noexcept -> std::optional<ImVec2>;

    /// Widget hovered during the current frame (only valid after begin_window calls).
    [[nodiscard]] auto cur_hovered_widget() const noexcept -> std::string_view;
    /// Widget hovered during the previous frame (stable — safe to read at any point).
    [[nodiscard]] auto prev_hovered_widget() const noexcept -> std::string_view;
    [[nodiscard]] auto cur_focused_widget() const noexcept -> std::string_view;

   private:
    void ensure_rendering_backend();
    void update_widget_tracking();

    rendering::IViewport& m_viewport;
    rendering::WebGpuContext& m_webgpu_context;
    LoggingManager& m_logging_manager;

    std::unique_ptr<rendering::IImguiWindowing> m_imgui_windowing;
    std::unique_ptr<rendering::IImguiRendering> m_imgui_rendering;

    std::unordered_map<std::string, WindowInfo> m_window_info;
    std::string m_cur_hovered_widget;
    std::string m_prev_hovered_widget;
    std::string m_cur_focused_widget;

    static constexpr auto k_no_hovered_widget = "";
};

}  // namespace pts
