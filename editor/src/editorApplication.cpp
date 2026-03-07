#include "editorApplication.h"

#include <core/commandLine.h>
#include <core/components/imguiComponent.h>
#include <core/components/inputComponent.h>
#include <core/imgui/fileDialogue.h>
#include <core/rendering/sceneLoader.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <core/rendering/webgpuContext.h>
#include <core/rendering/windowing.h>
#include <grid_shader_metadata.h>
#include <imgui_internal.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/usd/stage.h>
#include <shader_metadata.h>
#include <spdlog/sinks/ringbuffer_sink.h>

#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>

#include "editorResources.h"

using namespace pts;
using namespace pts::editor;

static constexpr auto k_scene_setting_win_name = "Scene Settings";
static constexpr auto k_inspector_win_name = "Inspector";
static constexpr auto k_scene_view_win_name = "Scene";
static constexpr auto k_console_win_name = "Console";
static constexpr auto k_console_log_buffer_size = 1024;

struct ForwardUniforms {
    glm::mat4 mvp;
    glm::mat4 model;
    glm::vec3 sun_dir;
    float time;
};
static_assert(sizeof(ForwardUniforms) == 144, "ForwardUniforms must match shader std140 layout");

struct GridUniforms {
    glm::mat4 inv_vp;
    glm::mat4 vp;
    glm::vec3 camera_pos;
    float near_plane;
    float far_plane;
    float _pad[3];
};
static_assert(sizeof(GridUniforms) == 160, "GridUniforms must match shader std140 layout");

EditorApplication::EditorApplication(std::string_view name, pts::LoggingManager& logging_manager)
    : WindowedApplication{name, logging_manager} {
    create_input_actions();

    m_console_log_sink =
        std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(k_console_log_buffer_size);
    get_logging_manager().add_sink(m_console_log_sink);

    log(pts::LogLevel::Info, "EditorApplication created");
}

EditorApplication::~EditorApplication() {
    m_input.reset();
    m_imgui.reset();

    if (m_grid_bind_group) {
        wgpuBindGroupRelease(m_grid_bind_group);
    }
    if (m_grid_bind_group_layout) {
        wgpuBindGroupLayoutRelease(m_grid_bind_group_layout);
    }
    if (m_bind_group) {
        wgpuBindGroupRelease(m_bind_group);
    }
    if (m_bind_group_layout) {
        wgpuBindGroupLayoutRelease(m_bind_group_layout);
    }
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
        auto stage = pxr::UsdStage::Open(layer);
        rendering::populate_from_stage(m_world, stage, device);
        log(LogLevel::Info, "Loaded default scene ({} objects)", m_world.objects.size());
    } else {
        log(LogLevel::Warning, "Missing embedded resource: scenes/default.usda");
    }

    // Load forward shader
    auto shader_src = editor_resources::get_resource("generated/shaders/forward.wgsl");
    if (!shader_src) {
        log(LogLevel::Error, "Missing embedded resource: generated/shaders/forward.wgsl");
        if (m_app_config.quit_on_start) viewport()->request_close();
        return;
    }
    m_forward_shader.emplace(device.create_shader_module_from_source(*shader_src));

    // Uniform buffer
    m_uniform_buffer = device.create_buffer(
        sizeof(ForwardUniforms),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    // Bind group layout from shader reflection
    m_bind_group_layout = editor_shader::create_bind_group_layout_0(device.handle());

    // Bind group
    WGPUBindGroupEntry bg_entry = WGPU_BIND_GROUP_ENTRY_INIT;
    bg_entry.binding = 0;
    bg_entry.buffer = m_uniform_buffer.handle();
    bg_entry.offset = 0;
    bg_entry.size = sizeof(ForwardUniforms);

    WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg_desc.layout = m_bind_group_layout;
    bg_desc.entryCount = 1;
    bg_desc.entries = &bg_entry;
    m_bind_group = wgpuDeviceCreateBindGroup(device.handle(), &bg_desc);

    // Pipeline layout
    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &m_bind_group_layout;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

    // Build render pipeline with depth testing and back-face culling
    m_forward_pipeline.emplace(webgpu::RenderPipelineBuilder(device)
                                   .shader(*m_forward_shader)
                                   .color_format(WGPUTextureFormat_RGBA8Unorm)
                                   .depth_format(WGPUTextureFormat_Depth24Plus)
                                   .depth_write(true)
                                   .depth_compare(WGPUCompareFunction_Less)
                                   .cull_mode(WGPUCullMode_Back)
                                   .pipeline_layout(pipeline_layout)
                                   .vertex_layout<editor_shader::VertexLayout>()
                                   .build());

    wgpuPipelineLayoutRelease(pipeline_layout);

    // ── Grid pipeline ──

    auto grid_shader_src = editor_resources::get_resource("generated/shaders/grid.wgsl");
    if (!grid_shader_src) {
        log(LogLevel::Error, "Missing embedded resource: generated/shaders/grid.wgsl");
    } else {
        m_grid_shader.emplace(device.create_shader_module_from_source(*grid_shader_src));

        // Grid uniform buffer
        m_grid_uniform_buffer = device.create_buffer(
            sizeof(GridUniforms),
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

        // Bind group layout from reflection
        m_grid_bind_group_layout = editor_grid_shader::create_bind_group_layout_0(device.handle());

        // Bind group
        WGPUBindGroupEntry grid_bg_entry = WGPU_BIND_GROUP_ENTRY_INIT;
        grid_bg_entry.binding = 0;
        grid_bg_entry.buffer = m_grid_uniform_buffer.handle();
        grid_bg_entry.offset = 0;
        grid_bg_entry.size = sizeof(GridUniforms);

        WGPUBindGroupDescriptor grid_bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        grid_bg_desc.layout = m_grid_bind_group_layout;
        grid_bg_desc.entryCount = 1;
        grid_bg_desc.entries = &grid_bg_entry;
        m_grid_bind_group = wgpuDeviceCreateBindGroup(device.handle(), &grid_bg_desc);

        // Pipeline layout
        WGPUPipelineLayoutDescriptor grid_pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        grid_pl_desc.bindGroupLayoutCount = 1;
        grid_pl_desc.bindGroupLayouts = &m_grid_bind_group_layout;
        WGPUPipelineLayout grid_pipeline_layout =
            wgpuDeviceCreatePipelineLayout(device.handle(), &grid_pl_desc);

        // Premultiplied alpha blending
        WGPUBlendState blend_state = {};
        blend_state.color.srcFactor = WGPUBlendFactor_One;
        blend_state.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
        blend_state.color.operation = WGPUBlendOperation_Add;
        blend_state.alpha.srcFactor = WGPUBlendFactor_One;
        blend_state.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
        blend_state.alpha.operation = WGPUBlendOperation_Add;

        m_grid_pipeline.emplace(webgpu::RenderPipelineBuilder(device)
                                    .shader(*m_grid_shader)
                                    .color_format(WGPUTextureFormat_RGBA8Unorm)
                                    .depth_format(WGPUTextureFormat_Depth24Plus)
                                    .depth_write(false)
                                    .depth_compare(WGPUCompareFunction_Less)
                                    .cull_mode(WGPUCullMode_None)
                                    .blend_state(blend_state)
                                    .pipeline_layout(grid_pipeline_layout)
                                    .build());

        wgpuPipelineLayoutRelease(grid_pipeline_layout);
    }

    // Camera defaults
    m_camera.set_target({0.0f, 0.0f, 0.0f});
    m_camera.set_distance(3.0f);
    m_camera.set_fov_y(60.0f);

    if (m_app_config.quit_on_start) {
        viewport()->request_close();
    }
}

void EditorApplication::update(float /*dt*/) {
    // Input polling and ImGui drawing happen in render() to ensure proper
    // synchronization with ImGui::NewFrame() and the FrameGraph.
}

void EditorApplication::render(FrameContext& ctx) {
    if (!m_imgui) return;
    if (viewport() && viewport()->should_close()) return;

    m_input->reset_scroll_delta();
    auto scope = m_imgui->frame_scope();

    m_input->poll(get_time(), window_width(), window_height(), m_imgui->cur_hovered_widget());

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

    rendering::TextureDesc surface_desc;
    surface_desc.width = ctx.width();
    surface_desc.height = ctx.height();
    surface_desc.format = ctx.surface_format();
    surface_desc.clear_color = {0.08, 0.08, 0.12, 1.0};
    auto surface = m_frame_graph->import("surface", ctx.surface_view(), surface_desc);

    rendering::ResourceHandle scene_color_handle;
    rendering::ResourceHandle scene_depth_handle;
    bool has_viewport =
        m_forward_pipeline.has_value() && m_viewport_width > 0 && m_viewport_height > 0;

    if (has_viewport) {
        float aspect = static_cast<float>(m_viewport_width) / static_cast<float>(m_viewport_height);
        auto view_mat = m_camera.view_matrix();
        auto proj_mat = m_camera.projection_matrix(aspect);

        rendering::TextureDesc color_desc;
        color_desc.width = m_viewport_width;
        color_desc.height = m_viewport_height;
        color_desc.format = WGPUTextureFormat_RGBA8Unorm;
        color_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                                         WGPUTextureUsage_TextureBinding);
        color_desc.clear_color = {0.15, 0.15, 0.18, 1.0};
        auto scene_color = m_frame_graph->create("scene_color", color_desc);

        rendering::TextureDesc depth_desc;
        depth_desc.width = m_viewport_width;
        depth_desc.height = m_viewport_height;
        depth_desc.format = WGPUTextureFormat_Depth24Plus;
        auto scene_depth = m_frame_graph->create("scene_depth", depth_desc);

        scene_color_handle = scene_color;
        scene_depth_handle = scene_depth;

        m_frame_graph->add_pass("forward")
            .color(scene_color)
            .depth(scene_depth)
            .execute([=](WGPURenderPassEncoder pass) {
                wgpuRenderPassEncoderSetPipeline(pass, m_forward_pipeline->handle());
                for (const auto& obj : m_world.objects) {
                    ForwardUniforms u;
                    u.mvp = proj_mat * view_mat * obj.transform;
                    u.model = obj.transform;
                    u.sun_dir = glm::normalize(glm::vec3(0.3f, 1.0f, 0.5f));
                    u.time = 0.0f;
                    wgpuQueueWriteBuffer(queue, m_uniform_buffer.handle(), 0, &u, sizeof(u));
                    wgpuRenderPassEncoderSetBindGroup(pass, 0, m_bind_group, 0, nullptr);
                    const auto& mesh = m_world.meshes[obj.mesh_index];
                    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh.vertex_buffer.handle(), 0,
                                                         mesh.vertex_buffer.size());
                    wgpuRenderPassEncoderSetIndexBuffer(pass, mesh.index_buffer.handle(),
                                                        WGPUIndexFormat_Uint32, 0,
                                                        mesh.index_buffer.size());
                    wgpuRenderPassEncoderDrawIndexed(pass, mesh.index_count, 1, 0, 0, 0);
                }
            });

        // Grid pass — renders after forward pass, reads depth, blends over color
        if (m_grid_pipeline.has_value()) {
            auto vp_mat = proj_mat * view_mat;
            auto inv_vp_mat = glm::inverse(vp_mat);
            auto cam_pos = m_camera.position();

            m_frame_graph->add_pass("grid")
                .color(scene_color)
                .depth_readonly(scene_depth)
                .execute([=](WGPURenderPassEncoder pass) {
                    GridUniforms gu;
                    gu.inv_vp = inv_vp_mat;
                    gu.vp = vp_mat;
                    gu.camera_pos = cam_pos;
                    gu.near_plane = m_camera.near_plane();
                    gu.far_plane = m_camera.far_plane();
                    gu._pad[0] = gu._pad[1] = gu._pad[2] = 0.0f;
                    wgpuQueueWriteBuffer(queue, m_grid_uniform_buffer.handle(), 0, &gu, sizeof(gu));
                    wgpuRenderPassEncoderSetPipeline(pass, m_grid_pipeline->handle());
                    wgpuRenderPassEncoderSetBindGroup(pass, 0, m_grid_bind_group, 0, nullptr);
                    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
                });
        }
    }

    // ImGui overlay pass
    m_frame_graph->add_pass("imgui").color(surface).present(surface).execute(
        [&](WGPURenderPassEncoder pass) { scope.render_into(pass); });

    m_frame_graph->compile();
    m_frame_graph->execute(ctx.encoder());

    // Store scene color view for next frame's ImGui::Image
    if (has_viewport && scene_color_handle.is_valid()) {
        m_scene_color_view = m_frame_graph->get_texture_view(scene_color_handle);
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
                m_world.clear();
                rendering::populate_from_stage(m_world, stage, webgpu_context()->device());
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
    ImGui::TextUnformatted("Scene system rewrite in progress.");
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

    if (m_scene_color_view && m_viewport_width > 0 && m_viewport_height > 0) {
        ImGui::Image(
            reinterpret_cast<ImTextureID>(m_scene_color_view),
            ImVec2(static_cast<float>(m_viewport_width), static_cast<float>(m_viewport_height)));
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

auto EditorApplication::on_mouse_leave_scene_viewport() noexcept -> void {
}

auto EditorApplication::on_mouse_enter_scene_viewport() noexcept -> void {
}

auto EditorApplication::handle_input(InputEvent const& event) noexcept -> void {
    if (event.initiated_window != k_scene_view_win_name) return;

    if (event.input.input_type == InputType::MOUSE) {
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
