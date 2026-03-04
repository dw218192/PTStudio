#include <core/commandLine.h>
#include <core/enumUtils.h>
#include <core/loggingManager.h>
#include <core/playground.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <core/rendering/webgpuContext.h>

#include <cstdio>
#include <exception>
#include <optional>

namespace {

constexpr auto k_shader_code = R"(
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2f, 3>(
        vec2f( 0.0,  0.5),
        vec2f(-0.5, -0.5),
        vec2f( 0.5, -0.5),
    );
    var colors = array<vec3f, 3>(
        vec3f(1.0, 0.0, 0.0),
        vec3f(0.0, 1.0, 0.0),
        vec3f(0.0, 0.0, 1.0),
    );

    var out: VertexOutput;
    out.position = vec4f(positions[idx], 0.0, 1.0);
    out.color = colors[idx];
    return out;
}

@fragment
fn fs_main(@location(0) color: vec3f) -> @location(0) vec4f {
    return vec4f(color, 1.0);
}
)";

}  // namespace

class HelloApp : public pts::Playground {
   public:
    explicit HelloApp(pts::LoggingManager& logging_manager)
        : pts::Playground({"Hello Triangle", 1280, 720}, logging_manager) {
    }

   private:
    std::optional<pts::webgpu::ShaderModule> m_shader;
    std::optional<pts::webgpu::RenderPipeline> m_pipeline;

    void on_ready() override {
        auto const& device = get_webgpu_context()->device();
        m_shader.emplace(device.create_shader_module_from_source(k_shader_code));
        m_pipeline.emplace(pts::webgpu::RenderPipelineBuilder(device)
                               .shader(*m_shader)
                               .color_format(get_webgpu_context()->surface_format())
                               .build());
    }

    void render(pts::FrameContext& ctx) override {
        ctx.render_pass({}, [&](WGPURenderPassEncoder pass) {
            wgpuRenderPassEncoderSetPipeline(pass, m_pipeline->handle());
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });
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
