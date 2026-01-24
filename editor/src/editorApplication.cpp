#include "editorApplication.h"

#include <core/imgui/fileDialogue.h>
#include <core/imgui/imhelper.h>
#include <core/loggingManager.h>
#include <imgui_internal.h>
#include <spdlog/sinks/ringbuffer_sink.h>

#include <cstring>
#include <filesystem>
#include <glm/gtc/type_ptr.hpp>

#include "editorResources.h"

using namespace pts;
using namespace pts::editor;

static constexpr auto k_scene_setting_win_name = "Scene Settings";
static constexpr auto k_inspector_win_name = "Inspector";
static constexpr auto k_scene_view_win_name = "Scene";
static constexpr auto k_console_win_name = "Console";
static constexpr auto k_console_log_buffer_size = 1024;

EditorApplication::EditorApplication(std::string_view name, RenderConfig config,
                                     AppConfig app_config, pts::LoggingManager& logging_manager,
                                     pts::PluginManager& plugin_manager)
    : GUIApplication{name,         logging_manager, plugin_manager,
                     config.width, config.height,   config.min_frame_time},
      m_app_config{app_config},
      m_config{config} {
    get_imgui_window_info(k_scene_view_win_name).on_enter_region.connect([this] {
        on_mouse_enter_scene_viewport();
    });
    get_imgui_window_info(k_scene_view_win_name).on_leave_region.connect([this] {
        on_mouse_leave_scene_viewport();
    });

    create_input_actions();

    m_console_log_sink =
        std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(k_console_log_buffer_size);
    get_logging_manager().add_sink(m_console_log_sink);

    m_renderer_host_api.render_graph_api = get_render_graph_api();
    m_renderer_host_api.render_world_api = nullptr;

    if (m_app_config.quit_on_start && m_viewport) {
        m_viewport->request_close();
    }

    m_renderer_plugin = get_plugin_manager().get_plugin_instance("editor.renderer");
    if (m_renderer_plugin) {
        m_renderer_interface =
            static_cast<RendererPluginInterfaceV1*>(get_plugin_manager().query_interface(
                m_renderer_plugin, RENDERER_PLUGIN_INTERFACE_V1_ID));
    }
    if (!m_renderer_interface) {
        log(pts::LogLevel::Error, "Renderer plugin interface not found");
    }

    log(pts::LogLevel::Info, "EditorApplication created (scene rewrite in progress)");
}

EditorApplication::~EditorApplication() = default;

auto EditorApplication::create_input_actions() noexcept -> void {
    m_input_actions.clear();
}

auto EditorApplication::wrap_mouse_pos() noexcept -> void {
}

auto EditorApplication::on_begin_first_loop() -> void {
    if (m_app_config.quit_on_start) {
        m_viewport->request_close();
    }

    GUIApplication::on_begin_first_loop();

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

auto EditorApplication::loop(float dt) -> void {
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                 ImGuiDockNodeFlags_PassthruCentralNode);
    if (m_renderer_interface && m_renderer_interface->build_graph) {
        PtsFrameParams frame{};
        frame.frame_index = m_frame_index++;
        frame.time_seconds = get_time();
        frame.wall_time = get_time();

        glm::mat4 view{1.0f};
        glm::mat4 proj{1.0f};
        auto view_proj = proj * view;

        PtsViewParams view_params{};
        std::memcpy(view_params.view, glm::value_ptr(view), sizeof(view_params.view));
        std::memcpy(view_params.proj, glm::value_ptr(proj), sizeof(view_params.proj));
        std::memcpy(view_params.view_proj, glm::value_ptr(view_proj),
                    sizeof(view_params.view_proj));
        std::memset(view_params.prev_view_proj, 0, sizeof(view_params.prev_view_proj));
        view_params.camera_pos[0] = 0.0f;
        view_params.camera_pos[1] = 0.0f;
        view_params.camera_pos[2] = 0.0f;
        view_params.jitter_xy[0] = 0.0f;
        view_params.jitter_xy[1] = 0.0f;
        view_params.dt_seconds = dt;
        view_params.frame_index = static_cast<uint32_t>(frame.frame_index);
        view_params.viewport_w = m_config.width;
        view_params.viewport_h = m_config.height;
        view_params.near_plane = 0.1f;
        view_params.far_plane = 100000.0f;

        PtsFrameIO io{};
        io.output = get_render_output_texture();

        set_render_graph_current();
        m_renderer_interface->build_graph(&m_renderer_host_api, &frame, &view_params, &io);
        clear_render_graph_current();
    }

    if (begin_imgui_window(k_scene_setting_win_name, ImGuiWindowFlags_NoMove)) {
        draw_scene_panel();
    }
    end_imgui_window();

    if (begin_imgui_window(k_inspector_win_name, ImGuiWindowFlags_NoMove)) {
        draw_object_panel();
    }
    end_imgui_window();

    if (begin_imgui_window(k_console_win_name, ImGuiWindowFlags_NoMove)) {
        draw_console_panel();
    }
    end_imgui_window();

    if (begin_imgui_window(k_scene_view_win_name, ImGuiWindowFlags_NoScrollWithMouse |
                                                      ImGuiWindowFlags_NoMove |
                                                      ImGuiWindowFlags_MenuBar)) {
        draw_scene_viewport();
    }
    end_imgui_window();

    wrap_mouse_pos();
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
        m_config.width = static_cast<unsigned>(view_size.x);
        m_config.height = static_cast<unsigned>(view_size.y);
        on_render_config_change(m_config);
        last_size = view_size;
    }

    auto output = get_render_output_imgui_id();
    if (output) {
        ImGui::Image(output, view_size);
    } else {
        ImGui::TextUnformatted("Renderer output not available");
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

auto EditorApplication::on_render_config_change(RenderConfig const& conf) -> void {
    resize_render_output(conf.width, conf.height);
    if (m_renderer_interface && m_renderer_interface->on_resize) {
        m_renderer_interface->on_resize(conf.width, conf.height);
    }
}

auto EditorApplication::on_mouse_leave_scene_viewport() noexcept -> void {
}

auto EditorApplication::on_mouse_enter_scene_viewport() noexcept -> void {
}

auto EditorApplication::handle_input(InputEvent const&) noexcept -> void {
}
