#include "editorApplication.h"

#include <core/commandLine.h>
#include <core/components/imguiComponent.h>
#include <core/components/inputComponent.h>
#include <imgui_internal.h>
#include <spdlog/sinks/ringbuffer_sink.h>

#include <filesystem>

#include "editorResources.h"

using namespace pts;
using namespace pts::editor;

static constexpr auto k_scene_setting_win_name = "Scene Settings";
static constexpr auto k_inspector_win_name = "Inspector";
static constexpr auto k_scene_view_win_name = "Scene";
static constexpr auto k_console_win_name = "Console";
static constexpr auto k_console_log_buffer_size = 1024;

EditorApplication::EditorApplication(std::string_view name,
                                     pts::LoggingManager& logging_manager)
    : WindowedApplication{name, logging_manager} {
    create_input_actions();

    m_console_log_sink =
        std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(k_console_log_buffer_size);
    get_logging_manager().add_sink(m_console_log_sink);

    log(pts::LogLevel::Info, "EditorApplication created (scene rewrite in progress)");
}

EditorApplication::~EditorApplication() {
    m_input.reset();
    m_imgui.reset();
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
    m_imgui = std::make_unique<ImGuiComponent>(*viewport(), *webgpu_context(),
                                               get_logging_manager());
    m_input = std::make_unique<InputComponent>(*viewport());
    m_input->set_handler([this](const InputEvent& e) { handle_input(e); });

    m_imgui->get_window_info(k_scene_view_win_name).on_enter_region.connect([this] {
        on_mouse_enter_scene_viewport();
    });
    m_imgui->get_window_info(k_scene_view_win_name).on_leave_region.connect([this] {
        on_mouse_leave_scene_viewport();
    });

    if (m_app_config.quit_on_start) {
        viewport()->request_close();
    }
}

void EditorApplication::update(float dt) {
    if (!m_imgui) return;

    m_input->reset_scroll_delta();
    m_imgui->begin_frame();

    m_input->poll(get_time(), window_width(), window_height(),
                  m_imgui->cur_hovered_widget());

    if (m_first_frame) {
        setup_docking_layout();
        m_first_frame = false;
    }

    if (viewport() && viewport()->should_close()) {
        return;
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

    wrap_mouse_pos();

    m_imgui->end_frame();
}

void EditorApplication::render(FrameContext& /*ctx*/) {
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
    ImGui::BeginDisabled();
    ImGui::Button("Open Scene");
    ImGui::SameLine();
    ImGui::Button("Save Scene");
    ImGui::EndDisabled();
}

auto EditorApplication::draw_object_panel() noexcept -> void {
    ImGui::TextUnformatted("Scene system rewrite in progress.");
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
        last_size = view_size;
    }

    ImGui::TextUnformatted("Renderer output not available");
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
auto EditorApplication::handle_input(InputEvent const&) noexcept -> void {
}
