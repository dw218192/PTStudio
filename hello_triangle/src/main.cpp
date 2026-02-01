#include <core/application.h>
#include <core/loggingManager.h>
#include <core/pluginManager.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <core/rendering/webgpu/webgpu.h>
#include <core/rendering/webgpuContext.h>

#include <optional>

class HelloTriangleApp : public pts::Application {
   public:
    HelloTriangleApp(pts::LoggingManager& logging_manager, pts::PluginManager& plugin_manager)
        : pts::Application("HelloTriangle", logging_manager, plugin_manager, 800, 600,
                           1.0f / 60.0f),
          m_pipeline(create_pipeline()) {
    }

    void loop(float dt) override {
        static_cast<void>(dt);

        auto* context = get_webgpu_context();
        if (!context) {
            return;
        }

        WGPUTextureView view = context->surface().acquire_texture_view();
        if (view == nullptr) {
            return;
        }

        WGPUCommandEncoderDescriptor encoder_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        WGPUCommandEncoder encoder =
            wgpuDeviceCreateCommandEncoder(context->device().handle(), &encoder_desc);
        if (encoder == nullptr) {
            context->surface().present();
            return;
        }

        WGPUColor clear_color{0.1, 0.2, 0.3, 1.0};
        WGPURenderPassColorAttachment color_attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        color_attachment.view = view;
        color_attachment.loadOp = WGPULoadOp_Clear;
        color_attachment.storeOp = WGPUStoreOp_Store;
        color_attachment.clearValue = clear_color;

        WGPURenderPassDescriptor pass_desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        pass_desc.colorAttachmentCount = 1;
        pass_desc.colorAttachments = &color_attachment;

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
        if (pass != nullptr) {
            wgpuRenderPassEncoderSetPipeline(pass, m_pipeline.handle());
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            wgpuRenderPassEncoderEnd(pass);
            wgpuRenderPassEncoderRelease(pass);
        }

        WGPUCommandBufferDescriptor cmd_desc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
        wgpuCommandEncoderRelease(encoder);
        if (cmd != nullptr) {
            wgpuQueueSubmit(context->device().queue(), 1, &cmd);
            wgpuCommandBufferRelease(cmd);
        }

        context->surface().present();
    }

   private:
    auto create_pipeline() -> pts::webgpu::RenderPipeline {
        auto* context = get_webgpu_context();
        if (!context) {
            throw std::runtime_error("WebGPU context not available");
        }

        constexpr const char* kShaderSource = R"(
@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4f {
    var positions = array<vec2f, 3>(
        vec2f(0.0, 0.5),
        vec2f(-0.5, -0.5),
        vec2f(0.5, -0.5)
    );
    return vec4f(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(1.0, 0.5, 0.2, 1.0);
}
)";

        auto shader = context->device().create_shader_module_from_source(kShaderSource);

        log(pts::LogLevel::Info, "Triangle pipeline initialized");

        return pts::webgpu::RenderPipelineBuilder(context->device())
            .shader(shader)
            .vertex_entry("vs_main")
            .fragment_entry("fs_main")
            .color_format(context->surface_format())
            .build();
    }

    pts::webgpu::RenderPipeline m_pipeline;
};

int main() {
    pts::Config config;
    config.level = pts::LogLevel::Info;
    config.pattern = "[%H:%M:%S] [%^%L%$] [%n] %v";

    pts::LoggingManager logging_manager(config);
    auto logger = logging_manager.get_logger_shared("HelloTriangle");
    pts::PluginManager plugin_manager(logger, logging_manager);

    try {
        HelloTriangleApp app(logging_manager, plugin_manager);
        app.run();
    } catch (const std::exception& e) {
        logging_manager.get_logger().error("Application error: {}", e.what());
        return 1;
    }

    return 0;
}
