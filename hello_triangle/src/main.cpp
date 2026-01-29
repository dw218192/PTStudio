#include <core/application.h>
#include <core/loggingManager.h>
#include <core/pluginManager.h>
#include <core/rendering/webgpu/webgpu.h>
#include <core/rendering/webgpuContext.h>

#include <cstring>
#include <memory>

class HelloTriangleApp : public pts::Application {
   public:
    HelloTriangleApp(pts::LoggingManager& logging_manager, pts::PluginManager& plugin_manager)
        : pts::Application("Hello Triangle", logging_manager, plugin_manager, 800, 600,
                          1.0f / 60.0f) {
        init_pipeline();
    }

    ~HelloTriangleApp() override {
        if (m_pipeline != nullptr) {
            wgpuRenderPipelineRelease(m_pipeline);
        }
    }

    void loop(float dt) override {
        static_cast<void>(dt);

        auto* context = get_webgpu_context();
        if (!context || !context->is_valid()) {
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
            if (m_pipeline != nullptr) {
                wgpuRenderPassEncoderSetPipeline(pass, m_pipeline);
                wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            }
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
    void init_pipeline() {
        auto* context = get_webgpu_context();
        if (!context || !context->is_valid()) {
            log(pts::LogLevel::Error, "WebGPU context not available");
            return;
        }

        const char* shader_source = R"(
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

        WGPUShaderSourceWGSL wgsl_descriptor = {};
        wgsl_descriptor.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgsl_descriptor.code = WGPUStringView{shader_source, std::strlen(shader_source)};

        WGPUShaderModuleDescriptor shader_desc = {};
        shader_desc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl_descriptor);
        WGPUShaderModule shader_module =
            wgpuDeviceCreateShaderModule(context->device().handle(), &shader_desc);
        if (shader_module == nullptr) {
            log(pts::LogLevel::Error, "Failed to create shader module");
            return;
        }

        WGPUColorTargetState color_target = {};
        color_target.format = context->surface_format();
        color_target.writeMask = WGPUColorWriteMask_All;

        WGPUFragmentState fragment_state = {};
        fragment_state.module = shader_module;
        fragment_state.entryPoint = WGPUStringView{"fs_main", 7};
        fragment_state.targetCount = 1;
        fragment_state.targets = &color_target;

        WGPUVertexState vertex_state = {};
        vertex_state.module = shader_module;
        vertex_state.entryPoint = WGPUStringView{"vs_main", 7};

        WGPURenderPipelineDescriptor pipeline_desc = {};
        pipeline_desc.vertex = vertex_state;
        pipeline_desc.fragment = &fragment_state;
        pipeline_desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;

        m_pipeline = wgpuDeviceCreateRenderPipeline(context->device().handle(), &pipeline_desc);
        wgpuShaderModuleRelease(shader_module);

        if (m_pipeline == nullptr) {
            log(pts::LogLevel::Error, "Failed to create render pipeline");
            return;
        }

        log(pts::LogLevel::Info, "Triangle pipeline initialized");
    }

    WGPURenderPipeline m_pipeline{nullptr};
};

int main() {
    pts::Config config;
    config.level = pts::LogLevel::Info;
    config.pattern = "[%H:%M:%S] [%^%L%$] [%n] %v";
    
    pts::LoggingManager logging_manager(config);
    auto logger = logging_manager.get_logger_shared("hello_triangle");
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
