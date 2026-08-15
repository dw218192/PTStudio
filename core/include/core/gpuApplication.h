#pragma once

#include <core/application.h>
#include <core/rendering/webgpu/webgpu.h>

#include <memory>

namespace pts {
namespace rendering {
class WebGpuContext;
class IWindowing;
class IViewport;
}  // namespace rendering

namespace webgpu {
class Device;
}

struct RenderPassDesc {
    WGPUColor clear_color = {0.1, 0.1, 0.1, 1.0};
    bool depth = false;
};

struct GpuApplication;

class FrameContext {
    NO_COPY_MOVE(FrameContext);
    friend struct GpuApplication;

   public:
    [[nodiscard]] auto device() const noexcept -> const webgpu::Device&;
    [[nodiscard]] auto encoder() const noexcept -> WGPUCommandEncoder {
        return m_encoder;
    }
    [[nodiscard]] auto surface_view() const noexcept -> WGPUTextureView {
        return m_surface_view;
    }
    [[nodiscard]] auto surface_format() const noexcept -> WGPUTextureFormat {
        return m_surface_format;
    }
    [[nodiscard]] auto width() const noexcept -> uint32_t {
        return m_width;
    }
    [[nodiscard]] auto height() const noexcept -> uint32_t {
        return m_height;
    }
    [[nodiscard]] auto depth_view() const noexcept -> WGPUTextureView {
        return m_depth_view;
    }

    template <typename Fn>
    void render_pass(const RenderPassDesc& desc, Fn&& fn) {
        WGPURenderPassColorAttachment color_attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        color_attachment.view = m_surface_view;
        color_attachment.loadOp = WGPULoadOp_Clear;
        color_attachment.storeOp = WGPUStoreOp_Store;
        color_attachment.clearValue = desc.clear_color;

        WGPURenderPassDepthStencilAttachment depth_attachment =
            WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
        if (desc.depth) {
            depth_attachment.view = m_depth_view;
            depth_attachment.depthLoadOp = WGPULoadOp_Clear;
            depth_attachment.depthStoreOp = WGPUStoreOp_Store;
            depth_attachment.depthClearValue = 1.0f;
        }

        WGPURenderPassDescriptor pass_desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        pass_desc.colorAttachmentCount = 1;
        pass_desc.colorAttachments = &color_attachment;
        if (desc.depth) {
            pass_desc.depthStencilAttachment = &depth_attachment;
        }

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(m_encoder, &pass_desc);
        fn(pass);
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
    }

   private:
    FrameContext(const webgpu::Device& device, WGPUCommandEncoder encoder,
                 WGPUTextureView surface_view, WGPUTextureFormat surface_format,
                 WGPUTextureView depth_view, uint32_t w, uint32_t h)
        : m_device{&device},
          m_encoder{encoder},
          m_surface_view{surface_view},
          m_surface_format{surface_format},
          m_depth_view{depth_view},
          m_width{w},
          m_height{h} {
    }

    const webgpu::Device* m_device;
    WGPUCommandEncoder m_encoder;
    WGPUTextureView m_surface_view;
    WGPUTextureFormat m_surface_format;
    WGPUTextureView m_depth_view;
    uint32_t m_width;
    uint32_t m_height;
};

/**
 * @brief GPU application with headless-first design.
 *
 * Owns the WebGPU device and render loop. Windowing is opt-in:
 * subclasses call init_windowing() to create a window + surface.
 * Without windowing, the application runs headless (no surface, no present).
 */
struct GpuApplication : Application {
    NO_COPY_MOVE(GpuApplication);

    explicit GpuApplication(std::string_view name, pts::LoggingManager& logging_manager);
    ~GpuApplication() override;

    void run() override;

   protected:
    virtual void on_ready() {
    }
    virtual void render(FrameContext& /*ctx*/) {
    }
    virtual void update(float /*dt*/) {
    }
    virtual void on_resize(uint32_t /*w*/, uint32_t /*h*/) {
    }

    /// Create windowing system, viewport, and attach surface to the WebGPU context.
    /// Must be called after WebGPU context is ready (i.e. from on_ready() or later).
    void init_windowing();

    [[nodiscard]] auto webgpu_context() noexcept -> rendering::WebGpuContext*;
    [[nodiscard]] auto webgpu_context() const noexcept -> const rendering::WebGpuContext*;
    [[nodiscard]] auto windowing() noexcept -> rendering::IWindowing*;
    [[nodiscard]] auto viewport() noexcept -> rendering::IViewport*;
    [[nodiscard]] auto viewport() const noexcept -> const rendering::IViewport*;
    [[nodiscard]] auto window_width() const noexcept -> int;
    [[nodiscard]] auto window_height() const noexcept -> int;

    void ensure_depth_buffer(uint32_t w, uint32_t h);
    [[nodiscard]] auto depth_view() const noexcept -> WGPUTextureView;

   private:
    void loop(float dt) final;
    [[nodiscard]] bool ensure_webgpu_ready();

    std::unique_ptr<rendering::IWindowing> m_windowing;
    std::unique_ptr<rendering::IViewport> m_viewport;
    std::unique_ptr<rendering::WebGpuContext> m_webgpu_context;

    WGPUTexture m_depth_texture = nullptr;
    WGPUTextureView m_depth_view = nullptr;
    uint32_t m_depth_width = 0;
    uint32_t m_depth_height = 0;

    bool m_ready = false;
    bool m_framebuffer_resized = false;
};

}  // namespace pts
