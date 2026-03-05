#include <core/rendering/renderGraph.h>
#include <core/rendering/renderWorld.h>

namespace pts::rendering {

void RenderGraph::add_pass(PassDesc desc, PassCallback callback) {
    m_passes.push_back({std::move(desc), std::move(callback)});
}

void RenderGraph::execute(WGPUCommandEncoder encoder, const RenderWorld& world) {
    for (auto& pass : m_passes) {
        WGPURenderPassColorAttachment color_attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        color_attachment.view = pass.desc.color_target;
        color_attachment.loadOp = WGPULoadOp_Clear;
        color_attachment.storeOp = WGPUStoreOp_Store;
        color_attachment.clearValue = pass.desc.clear_color;

        WGPURenderPassDepthStencilAttachment depth_attachment =
            WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
        if (pass.desc.depth_target) {
            depth_attachment.view = pass.desc.depth_target;
            depth_attachment.depthLoadOp = WGPULoadOp_Clear;
            depth_attachment.depthStoreOp = WGPUStoreOp_Store;
            depth_attachment.depthClearValue = 1.0f;
        }

        WGPURenderPassDescriptor pass_desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        pass_desc.colorAttachmentCount = 1;
        pass_desc.colorAttachments = &color_attachment;
        if (pass.desc.depth_target) {
            pass_desc.depthStencilAttachment = &depth_attachment;
        }

        WGPURenderPassEncoder pass_encoder = wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
        pass.callback(pass_encoder, world);
        wgpuRenderPassEncoderEnd(pass_encoder);
        wgpuRenderPassEncoderRelease(pass_encoder);
    }
}

void RenderGraph::clear() {
    m_passes.clear();
}

}  // namespace pts::rendering
