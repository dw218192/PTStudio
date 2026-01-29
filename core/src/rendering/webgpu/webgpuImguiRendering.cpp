#include "webgpuImguiRendering.h"

#include <imgui.h>
#include <imgui_impl_wgpu.h>

#include <memory>

#ifndef IMGUI_IMPL_WEBGPU_BACKEND_DAWN
#define IMGUI_IMPL_WEBGPU_BACKEND_DAWN
#endif

namespace pts::rendering {
WebGpuImguiRendering::WebGpuImguiRendering(WebGpuContext& context, IViewport& viewport,
                                           pts::LoggingManager& logging_manager)
    : m_context(&context),
      m_viewport(&viewport),
      m_logger(logging_manager.get_logger_shared("WebGpuImguiRendering")) {
    if (!m_logger) {
        throw std::runtime_error("Failed to create logger");
    }

    if (!m_context->is_valid()) {
        m_logger->error("Invalid WebGPU context provided");
        throw std::runtime_error("Invalid WebGPU context");
    }

    m_extent = viewport.drawable_extent();

    ImGui_ImplWGPU_InitInfo init_info{};
    init_info.Device = m_context->device().handle();
    init_info.NumFramesInFlight = 3;
    init_info.RenderTargetFormat = m_context->surface_format();
    init_info.DepthStencilFormat = WGPUTextureFormat_Undefined;

    if (!ImGui_ImplWGPU_Init(&init_info)) {
        m_logger->error("ImGui WebGPU backend initialization failed");
        throw std::runtime_error("ImGui WebGPU backend initialization failed");
    }

    m_logger->info("WebGPU ImGui backend initialized");
}

WebGpuImguiRendering::~WebGpuImguiRendering() {
    ImGui_ImplWGPU_Shutdown();
    m_logger->info("WebGPU ImGui backend destroyed");
}

void WebGpuImguiRendering::new_frame() {
    ImGui_ImplWGPU_NewFrame();
}

void WebGpuImguiRendering::render(bool framebuffer_resized) {
    auto const extent = m_viewport->drawable_extent();
    if (framebuffer_resized || extent.w != m_extent.w || extent.h != m_extent.h) {
        m_extent = extent;
        m_context->surface().resize(m_extent);
    }

    WGPUTextureView view = m_context->surface().acquire_texture_view();
    if (view == nullptr) {
        m_logger->warn("Failed to acquire surface texture view");
        return;
    }

    WGPUCommandEncoderDescriptor encoder_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder encoder =
        wgpuDeviceCreateCommandEncoder(m_context->device().handle(), &encoder_desc);
    if (encoder == nullptr) {
        m_logger->error("Failed to create command encoder");
        m_context->surface().present();
        return;
    }

    WGPUColor clear_color{0.08, 0.08, 0.12, 1.0};
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
        // Render ImGui on top
        ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
    } else {
        m_logger->error("Failed to begin render pass");
    }

    WGPUCommandBufferDescriptor cmd_desc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuCommandEncoderRelease(encoder);
    if (cmd != nullptr) {
        WGPUQueue queue = m_context->device().queue();
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
    } else {
        m_logger->error("Failed to finish command buffer");
    }

    m_context->surface().present();
}

void WebGpuImguiRendering::resize() {
    m_extent = m_viewport->drawable_extent();
    m_context->surface().resize(m_extent);
}

auto WebGpuImguiRendering::set_render_output(IRenderGraph& render_graph) -> ImTextureID {
    static_cast<void>(render_graph);
    return ImTextureID_Invalid;
}

void WebGpuImguiRendering::clear_render_output() {
}

auto WebGpuImguiRendering::output_id() const noexcept -> ImTextureID {
    return ImTextureID_Invalid;
}

void WebGpuImguiRendering::render_before_imgui() {
}

std::unique_ptr<IImguiRendering> create_webgpu_imgui_rendering(
    WebGpuContext& context, IViewport& viewport, pts::LoggingManager& logging_manager) {
    try {
        auto ptr = new WebGpuImguiRendering(context, viewport, logging_manager);
        return std::unique_ptr<IImguiRendering>(ptr);
    } catch (const std::runtime_error& e) {
        logging_manager.get_logger().error("Failed to create WebGPU imgui rendering: {}", e.what());
        return nullptr;
    }
}
}  // namespace pts::rendering