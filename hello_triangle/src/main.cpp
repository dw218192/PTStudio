#include <core/commandLine.h>
#include <core/components/imguiComponent.h>
#include <core/enumUtils.h>
#include <core/gpuApplication.h>
#include <core/loggingManager.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <core/rendering/shaderCompiler.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <core/rendering/webgpuContext.h>
#include <embedded_resources.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/usd/stage.h>
#include <shader_metadata.h>

#include <cstdio>
#include <exception>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <optional>
#include <stdexcept>

struct Uniforms {
    glm::mat4 mvp;
    float time;
    float rotation;
    float _pad[2];  // align to 16 bytes (std140)
};

class HelloApp : public pts::GpuApplication {
   public:
    explicit HelloApp(pts::LoggingManager& logging_manager)
        : pts::GpuApplication("Hello Triangle", logging_manager) {
    }

    ~HelloApp() override {
        m_imgui.reset();
        if (m_bind_group) {
            wgpuBindGroupRelease(m_bind_group);
        }
        if (m_bind_group_layout) {
            wgpuBindGroupLayoutRelease(m_bind_group_layout);
        }
    }

   private:
    pts::rendering::RenderWorld m_world;
    std::unique_ptr<pts::rendering::ShaderLoader> m_shader_loader;
    std::unique_ptr<pts::rendering::IShaderCompiler> m_shader_compiler;
    std::unique_ptr<pts::rendering::FrameGraph> m_graph;
    std::optional<pts::webgpu::ShaderModule> m_shader;
    std::optional<pts::webgpu::RenderPipeline> m_pipeline;
    pts::webgpu::Buffer m_uniform_buffer;
    WGPUBindGroup m_bind_group = nullptr;
    WGPUBindGroupLayout m_bind_group_layout = nullptr;

    std::unique_ptr<pts::ImGuiComponent> m_imgui;
    float m_time_scale = 1.0f;
    float m_rotation_speed = 1.0f;

    void on_ready() override {
        init_windowing();
        auto const& device = webgpu_context()->device();

        auto usda = hello_triangle_resources::get_resource("scenes/triangle.usda");
        if (!usda) {
            throw std::runtime_error("missing embedded resource: scenes/triangle.usda");
        }

        // Load USD stage from embedded resource
        auto layer = pxr::SdfLayer::CreateAnonymous(".usda");
        layer->ImportFromString(std::string{*usda});
        auto stage = pxr::UsdStage::Open(layer);
        pts::rendering::populate_from_stage(m_world, stage);
        m_world.upload_all_meshes(device);

        // Route WGSL through IShaderCompiler -- consistent with renderer passes.
        m_shader_loader = std::make_unique<pts::rendering::ShaderLoader>(
            get_logging_manager().get_logger_shared("shader_loader"));
        m_shader_loader->register_shader(
            "generated/shaders/hello_triangle.wgsl", "hello_triangle/shaders/hello_triangle.slang",
            "generated/shaders/hello_triangle.wgsl", hello_triangle_resources::get_resource);
        m_shader_compiler = pts::rendering::make_shader_compiler(*m_shader_loader);
        auto shader_wgsl = m_shader_compiler->compile(
            pts::rendering::ShaderKey{"generated/shaders/hello_triangle.wgsl"});
        m_shader.emplace(device.create_shader_module_from_source(shader_wgsl));

        // Create uniform buffer
        m_uniform_buffer = device.create_buffer(sizeof(Uniforms),
                                                WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);

        // Create bind group layout from shader reflection metadata
        m_bind_group_layout = hello_triangle_shader::create_bind_group_layout_0(device.handle());

        // Create bind group
        WGPUBindGroupEntry bg_entry = WGPU_BIND_GROUP_ENTRY_INIT;
        bg_entry.binding = 0;
        bg_entry.buffer = m_uniform_buffer.handle();
        bg_entry.offset = 0;
        bg_entry.size = sizeof(Uniforms);

        WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bg_desc.layout = m_bind_group_layout;
        bg_desc.entryCount = 1;
        bg_desc.entries = &bg_entry;
        m_bind_group = wgpuDeviceCreateBindGroup(device.handle(), &bg_desc);

        // Create pipeline layout
        WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        pl_desc.bindGroupLayoutCount = 1;
        pl_desc.bindGroupLayouts = &m_bind_group_layout;
        WGPUPipelineLayout pipeline_layout =
            wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

        // Build render pipeline using shader reflection metadata
        m_pipeline.emplace(pts::webgpu::RenderPipelineBuilder(device)
                               .shader(*m_shader)
                               .color_format(webgpu_context()->surface_format())
                               .pipeline_layout(pipeline_layout)
                               .vertex_layout<hello_triangle_shader::VertexLayout>()
                               .build());

        wgpuPipelineLayoutRelease(pipeline_layout);

        // Initialize ImGui
        m_imgui = std::make_unique<pts::ImGuiComponent>(*viewport(), *webgpu_context(),
                                                        get_logging_manager());

        // Initialize FrameGraph
        m_graph = std::make_unique<pts::rendering::FrameGraph>(
            device, get_logging_manager().get_logger_shared("frame_graph"));
    }

    void render(pts::FrameContext& ctx) override {
        if (!m_imgui) return;
        if (ctx.width() == 0 || ctx.height() == 0) return;

        // Begin ImGui frame (scope ensures end_frame is always called)
        auto scope = m_imgui->frame_scope();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(250, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Controls");
        ImGui::SliderFloat("Time Scale", &m_time_scale, 0.0f, 5.0f);
        ImGui::SliderFloat("Rotation Speed", &m_rotation_speed, 0.0f, 5.0f);
        ImGui::End();

        // Build frame graph
        auto const& device = webgpu_context()->device();
        float aspect = static_cast<float>(ctx.width()) / static_cast<float>(ctx.height());
        auto proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        auto view = glm::lookAt(glm::vec3(0, 0, 2), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        auto vp = proj * view;
        float t = get_time();

        m_graph->begin_frame();

        // 3D render pass (clears surface)
        m_graph->add_pass("forward")
            .color(ctx.surface_view(), WGPUColor{0.1, 0.1, 0.1, 1.0})
            .present()
            .execute([&](pts::rendering::ExecuteContext&, WGPURenderPassEncoder pass) {
                wgpuRenderPassEncoderSetPipeline(pass, m_pipeline->handle());
                auto objects = m_world.get_objects().span_raw();
                auto meshes = m_world.get_meshes().span_raw();
                for (const auto& entry : objects) {
                    if (!entry.active) continue;
                    if (!entry.value.visible) continue;
                    Uniforms uniforms;
                    uniforms.mvp = vp * entry.value.transform;
                    uniforms.time = t * m_time_scale;
                    uniforms.rotation = t * m_rotation_speed;
                    wgpuQueueWriteBuffer(device.queue(), m_uniform_buffer.handle(), 0, &uniforms,
                                         sizeof(uniforms));
                    wgpuRenderPassEncoderSetBindGroup(pass, 0, m_bind_group, 0, nullptr);

                    const auto& mesh = meshes[entry.value.mesh_index].value;
                    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh.vertex_buffer.handle(), 0,
                                                         mesh.vertex_buffer.size());
                    wgpuRenderPassEncoderSetIndexBuffer(pass, mesh.index_buffer.handle(),
                                                        WGPUIndexFormat_Uint32, 0,
                                                        mesh.index_buffer.size());
                    wgpuRenderPassEncoderDrawIndexed(pass, mesh.index_count, 1, 0, 0, 0);
                }
            });

        // ImGui overlay pass (preserves 3D content via Load)
        m_graph->add_pass("imgui")
            .color(ctx.surface_view())
            .execute([&](pts::rendering::ExecuteContext&, WGPURenderPassEncoder pass) {
                scope.render_into(pass);
            });

        m_graph->compile();
        m_graph->execute(ctx.encoder());
    }
};

int main(int argc, char* argv[]) {
    pts::CommandLine pre_cli;
    pre_cli.add_string("log-level", "Log level (trace, debug, info, warn, error, critical)");
    if (!pre_cli.parse(argc, argv)) {
        return 0;
    }

    auto log_level_str = pre_cli.get_string("log-level", "info");
    auto opt_log_level = pts::from_string<pts::LogLevel>(log_level_str);
    if (!opt_log_level) {
        std::fprintf(stderr, "Invalid log level: %s\n", log_level_str.c_str());
        return 1;
    }

    pts::Config config;
    config.level = *opt_log_level;
    config.pattern = "[%H:%M:%S] [%^%L%$] [%n] %v";

    pts::LoggingManager logging_manager(config);

    try {
        HelloApp app(logging_manager);
        if (auto exit_code = app.init(argc, argv)) {
            return *exit_code;
        }
        app.run();
    } catch (const std::exception& e) {
        logging_manager.get_logger().error("Application error: {}", e.what());
        return 1;
    }

    return 0;
}
