#include <core/diagnostics.h>
#include <core/gpuApplication.h>
#include <core/rendering/webgpuContext.h>

namespace pts {

auto FrameContext::device() const noexcept -> const webgpu::Device& {
    return *m_device;
}

GpuApplication::GpuApplication(std::string_view name, pts::LoggingManager& logging_manager)
    : Application{name, logging_manager} {
}

void GpuApplication::init_windowing() {
    PRECONDITION_MSG(
        m_webgpu_context != nullptr && m_webgpu_context->is<rendering::ContextReadyState>(),
        "init_windowing() requires ready WebGPU context");
    PRECONDITION_MSG(!m_windowing, "init_windowing() already called");

    m_windowing = pts::rendering::create_windowing(get_logging_manager());
    INVARIANT_MSG(m_windowing != nullptr, "create_windowing must return valid windowing system");

    auto viewport_desc = pts::rendering::ViewportDesc{
        get_name().data(), m_width, m_height, true, true, true, true,
    };
    m_viewport = m_windowing->create_viewport(viewport_desc);
    INVARIANT_MSG(m_viewport != nullptr, "create_viewport must return valid viewport");
    m_viewport->on_drawable_resized.connect(
        [this](pts::rendering::Extent2D) { m_framebuffer_resized = true; });

    m_webgpu_context->create_surface(*m_viewport);
}

GpuApplication::~GpuApplication() {
    if (m_depth_view) {
        wgpuTextureViewRelease(m_depth_view);
    }
    if (m_depth_texture) {
        wgpuTextureDestroy(m_depth_texture);
        wgpuTextureRelease(m_depth_texture);
    }
}

void GpuApplication::run() {
    m_webgpu_context = pts::rendering::WebGpuContext::create_headless(get_logging_manager());
    INVARIANT_MSG(m_webgpu_context != nullptr,
                  "WebGpuContext::create_headless must return valid context");

#if defined(__EMSCRIPTEN__)
    Application::run();
#else
    while (!should_stop()) {
        if (m_viewport && m_viewport->should_close()) break;
        run_one_frame();
        check_frame_limit();
    }
#endif
}

bool GpuApplication::ensure_webgpu_ready() {
    if (m_webgpu_context->is<rendering::ContextFailedState>()) {
        return false;
    }

    if (m_webgpu_context->is<rendering::ContextInitializingState>()) {
        m_webgpu_context->tick();

        if (m_webgpu_context->is<rendering::ContextFailedState>()) {
            log(pts::LogLevel::Error, "WebGPU context initialization failed");
            if (m_viewport) {
                m_viewport->request_close();
            } else {
                request_stop();
            }
            return false;
        }

        if (m_webgpu_context->is<rendering::ContextInitializingState>()) {
            return false;
        }

        log(pts::LogLevel::Info, "Application initialized");
    }

    return true;
}

void GpuApplication::ensure_depth_buffer(uint32_t w, uint32_t h) {
    if (m_depth_width == w && m_depth_height == h) {
        return;
    }

    if (m_depth_view) {
        wgpuTextureViewRelease(m_depth_view);
        m_depth_view = nullptr;
    }
    if (m_depth_texture) {
        wgpuTextureDestroy(m_depth_texture);
        wgpuTextureRelease(m_depth_texture);
        m_depth_texture = nullptr;
    }

    auto const& device = m_webgpu_context->device();

    WGPUTextureDescriptor tex_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    tex_desc.size = {w, h, 1};
    tex_desc.format = WGPUTextureFormat_Depth32Float;
    tex_desc.usage = WGPUTextureUsage_RenderAttachment;
    tex_desc.mipLevelCount = 1;
    tex_desc.sampleCount = 1;
    tex_desc.dimension = WGPUTextureDimension_2D;
    m_depth_texture = wgpuDeviceCreateTexture(device.handle(), &tex_desc);

    WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    view_desc.format = WGPUTextureFormat_Depth32Float;
    view_desc.dimension = WGPUTextureViewDimension_2D;
    view_desc.mipLevelCount = 1;
    view_desc.arrayLayerCount = 1;
    m_depth_view = wgpuTextureCreateView(m_depth_texture, &view_desc);

    m_depth_width = w;
    m_depth_height = h;
}

void GpuApplication::loop(float dt) {
    if (m_windowing) {
        m_windowing->pump_events(pts::rendering::PumpEventMode::Poll);
    }

    if (!ensure_webgpu_ready()) {
        return;
    }

    if (!m_ready) {
        on_ready();
        m_ready = true;
    }

    if (m_viewport && m_framebuffer_resized && m_webgpu_context->has_surface()) {
        auto const extent = m_viewport->drawable_extent();
        m_webgpu_context->surface().resize(extent);
        on_resize(extent.w, extent.h);
        m_framebuffer_resized = false;
    }

    update(dt);

    // Determine render dimensions and surface view
    WGPUTextureView surface_view = nullptr;
    WGPUTextureFormat surface_fmt = WGPUTextureFormat_BGRA8Unorm;
    uint32_t render_w = m_width;
    uint32_t render_h = m_height;

    if (m_viewport && m_webgpu_context->has_surface()) {
        auto& surface = m_webgpu_context->surface();
        surface_view = surface.acquire_texture_view();
        if (!surface_view) {
            return;
        }
        auto const extent = m_viewport->drawable_extent();
        render_w = extent.w;
        render_h = extent.h;
        surface_fmt = m_webgpu_context->surface_format();
    }

    ensure_depth_buffer(render_w, render_h);

    auto const& device = m_webgpu_context->device();

    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    enc_desc.label = {"frame_encoder", WGPU_STRLEN};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device.handle(), &enc_desc);

    FrameContext ctx{device, encoder, surface_view, surface_fmt, m_depth_view, render_w, render_h};
    render(ctx);

    WGPUCommandBufferDescriptor cmd_desc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(device.queue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    if (surface_view && m_webgpu_context->has_surface()) {
        m_webgpu_context->surface().present();
    }
}

auto GpuApplication::webgpu_context() noexcept -> rendering::WebGpuContext* {
    return m_webgpu_context.get();
}

auto GpuApplication::webgpu_context() const noexcept -> const rendering::WebGpuContext* {
    return m_webgpu_context.get();
}

auto GpuApplication::windowing() noexcept -> rendering::IWindowing* {
    return m_windowing.get();
}

auto GpuApplication::viewport() noexcept -> rendering::IViewport* {
    return m_viewport.get();
}

auto GpuApplication::viewport() const noexcept -> const rendering::IViewport* {
    return m_viewport.get();
}

auto GpuApplication::window_width() const noexcept -> int {
    if (!m_viewport) return static_cast<int>(m_width);
    return static_cast<int>(m_viewport->drawable_extent().w);
}

auto GpuApplication::window_height() const noexcept -> int {
    if (!m_viewport) return static_cast<int>(m_height);
    return static_cast<int>(m_viewport->drawable_extent().h);
}

auto GpuApplication::depth_view() const noexcept -> WGPUTextureView {
    return m_depth_view;
}

}  // namespace pts
