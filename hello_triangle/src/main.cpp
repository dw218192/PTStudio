#include <core/commandLine.h>
#include <core/enumUtils.h>
#include <core/loggingManager.h>
#include <core/playground.h>
#include <core/rendering/renderGraph.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <core/rendering/webgpuContext.h>
#include <embedded_resources.h>
#include <shader_metadata.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/usd/stage.h>

#include <cstdio>
#include <exception>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <optional>
#include <stdexcept>

class HelloApp : public pts::Playground {
   public:
    explicit HelloApp(pts::LoggingManager& logging_manager)
        : pts::Playground({"Hello Triangle", 1280, 720}, logging_manager) {
    }

    ~HelloApp() override {
        if (m_bind_group) {
            wgpuBindGroupRelease(m_bind_group);
        }
        if (m_bind_group_layout) {
            wgpuBindGroupLayoutRelease(m_bind_group_layout);
        }
    }

   private:
    pts::rendering::RenderWorld m_world;
    pts::rendering::RenderGraph m_graph;
    std::optional<pts::webgpu::ShaderModule> m_shader;
    std::optional<pts::webgpu::RenderPipeline> m_pipeline;
    pts::webgpu::Buffer m_uniform_buffer;
    WGPUBindGroup m_bind_group = nullptr;
    WGPUBindGroupLayout m_bind_group_layout = nullptr;

    void on_ready() override {
        auto const& device = get_webgpu_context()->device();

        auto usda = hello_triangle_resources::get_resource("scenes/triangle.usda");
        if (!usda) {
            throw std::runtime_error("missing embedded resource: scenes/triangle.usda");
        }
        auto shader_src =
            hello_triangle_resources::get_resource("generated/shaders/hello_triangle.wgsl");
        if (!shader_src) {
            throw std::runtime_error(
                "missing embedded resource: generated/shaders/hello_triangle.wgsl");
        }

        // Load USD stage from embedded resource
        auto layer = pxr::SdfLayer::CreateAnonymous(".usda");
        layer->ImportFromString(std::string{*usda});
        auto stage = pxr::UsdStage::Open(layer);
        pts::rendering::populate_from_stage(m_world, stage, device);

        // Create shader module
        m_shader.emplace(device.create_shader_module_from_source(*shader_src));

        // Create uniform buffer for MVP matrix
        m_uniform_buffer = device.create_buffer(sizeof(glm::mat4),
                                                WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);

        // Create bind group layout from shader reflection metadata
        m_bind_group_layout =
            hello_triangle_shader::create_bind_group_layout_0(device.handle());

        // Create bind group
        WGPUBindGroupEntry bg_entry = WGPU_BIND_GROUP_ENTRY_INIT;
        bg_entry.binding = 0;
        bg_entry.buffer = m_uniform_buffer.handle();
        bg_entry.offset = 0;
        bg_entry.size = sizeof(glm::mat4);

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
                               .color_format(get_webgpu_context()->surface_format())
                               .pipeline_layout(pipeline_layout)
                               .vertex_layout<hello_triangle_shader::VertexLayout>()
                               .build());

        wgpuPipelineLayoutRelease(pipeline_layout);
    }

    void render(pts::FrameContext& ctx) override {
        auto const& device = get_webgpu_context()->device();

        // Compute MVP
        float aspect = static_cast<float>(ctx.width()) / static_cast<float>(ctx.height());
        auto proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        auto view = glm::lookAt(glm::vec3(0, 0, 2), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        auto vp = proj * view;

        // Write MVP for each object and draw via RenderGraph
        m_graph.clear();
        m_graph.add_pass(
            {"forward", ctx.surface_view(), ctx.surface_format()},
            [&](WGPURenderPassEncoder pass, const pts::rendering::RenderWorld& world) {
                wgpuRenderPassEncoderSetPipeline(pass, m_pipeline->handle());
                for (const auto& obj : world.objects) {
                    auto mvp = vp * obj.transform;
                    wgpuQueueWriteBuffer(device.queue(), m_uniform_buffer.handle(), 0, &mvp,
                                         sizeof(mvp));
                    wgpuRenderPassEncoderSetBindGroup(pass, 0, m_bind_group, 0, nullptr);

                    const auto& mesh = world.meshes[obj.mesh_index];
                    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh.vertex_buffer.handle(), 0,
                                                         mesh.vertex_buffer.size());
                    wgpuRenderPassEncoderSetIndexBuffer(pass, mesh.index_buffer.handle(),
                                                        WGPUIndexFormat_Uint32, 0,
                                                        mesh.index_buffer.size());
                    wgpuRenderPassEncoderDrawIndexed(pass, mesh.index_count, 1, 0, 0, 0);
                }
            });
        m_graph.execute(ctx.encoder(), m_world);
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
        if (!app.init(argc, argv)) {
            return 0;
        }
        app.run();
    } catch (const std::exception& e) {
        logging_manager.get_logger().error("Application error: {}", e.what());
        return 1;
    }

    return 0;
}
