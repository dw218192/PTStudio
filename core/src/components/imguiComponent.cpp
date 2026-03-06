#include <core/components/imguiComponent.h>
#include <core/diagnostics.h>
#include <core/rendering/webgpuContext.h>
#include <core/rendering/windowing.h>
#include <imgui_internal.h>

#include "../rendering/imguiBackend.h"
#include "../rendering/renderingComponents.h"

namespace pts {

ImGuiComponent::ImGuiComponent(rendering::IViewport& viewport,
                               rendering::WebGpuContext& webgpu_context,
                               LoggingManager& logging_manager)
    : m_viewport{viewport}, m_webgpu_context{webgpu_context}, m_logging_manager{logging_manager} {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();

    m_imgui_windowing = rendering::create_imgui_windowing(m_viewport, m_logging_manager);
}

ImGuiComponent::~ImGuiComponent() {
    m_imgui_rendering.reset();
    m_imgui_windowing.reset();
    ImGui::DestroyContext();
}

void ImGuiComponent::ensure_rendering_backend() {
    if (m_imgui_rendering) return;

    INVARIANT_MSG(m_webgpu_context.is_ready(), "WebGPU context must be ready");

    auto imgui_components =
        rendering::create_imgui_components(m_webgpu_context, m_viewport, m_logging_manager);
    INVARIANT_MSG(imgui_components.imgui_rendering != nullptr,
                  "create_imgui_components must return valid imgui_rendering");

    m_imgui_rendering = std::move(imgui_components.imgui_rendering);
}

void ImGuiComponent::begin_frame() {
    ensure_rendering_backend();

    m_prev_hovered_widget = m_cur_hovered_widget;
    m_cur_hovered_widget = "";
    m_cur_focused_widget = "";

    m_imgui_windowing->new_frame();
    m_imgui_rendering->new_frame();
    ImGui::NewFrame();
}

void ImGuiComponent::end_frame() {
    ImGui::Render();
    m_imgui_rendering->render(false);

    if (m_prev_hovered_widget != m_cur_hovered_widget) {
        if (m_prev_hovered_widget != k_no_hovered_widget) {
            auto it = m_window_info.find(m_prev_hovered_widget);
            if (it != m_window_info.end()) {
                it->second.on_leave_region();
            }
        }

        auto it = m_window_info.find(m_cur_hovered_widget);
        if (it != m_window_info.end()) {
            it->second.on_enter_region();
        }
    }
}

bool ImGuiComponent::is_ready() const noexcept {
    return m_imgui_rendering != nullptr;
}

auto ImGuiComponent::get_window_info(std::string_view name) noexcept -> WindowInfo& {
    return m_window_info[name];
}

auto ImGuiComponent::begin_window(std::string_view name, ImGuiWindowFlags flags) noexcept -> bool {
    auto const ret = ImGui::Begin(name.data(), nullptr, flags);
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
        m_cur_hovered_widget = name;
    }
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows)) {
        m_cur_focused_widget = name;
    }
    return ret;
}

void ImGuiComponent::end_window() noexcept {
    ImGui::End();
}

auto ImGuiComponent::get_window_content_pos(std::string_view name) const noexcept
    -> std::optional<ImVec2> {
    auto const win = ImGui::FindWindowByName(name.data());
    if (!win) {
        return std::nullopt;
    }
    return win->ContentRegionRect.Min;
}

auto ImGuiComponent::cur_hovered_widget() const noexcept -> std::string_view {
    return m_cur_hovered_widget;
}

auto ImGuiComponent::cur_focused_widget() const noexcept -> std::string_view {
    return m_cur_focused_widget;
}

}  // namespace pts
