#include <core/diagnostics.h>
#include <core/rendering/webgpuContext.h>
#include <core/windowedApplication.h>

namespace pts {

auto FrameContext::device() const noexcept -> const webgpu::Device& {
    return *m_device;
}

WindowedApplication::WindowedApplication(std::string_view name,
                                         pts::LoggingManager& logging_manager)
    : Application{name, logging_manager} {
}

void WindowedApplication::init_windowing() {
    if (m_windowing) return;

    m_windowing = pts::rendering::create_windowing(get_logging_manager());
    INVARIANT_MSG(m_windowing != nullptr, "create_windowing must return valid windowing system");

    auto viewport_desc = pts::rendering::ViewportDesc{
        get_name().data(), m_width, m_height, true, true, true, true,
    };
    m_viewport = m_windowing->create_viewport(viewport_desc);
    INVARIANT_MSG(m_viewport != nullptr, "create_viewport must return valid viewport");
    m_viewport->on_drawable_resized.connect(
        [this](pts::rendering::Extent2D) { m_framebuffer_resized = true; });

    m_webgpu_context = pts::rendering::WebGpuContext::create(*m_viewport, get_logging_manager());
    INVARIANT_MSG(m_webgpu_context != nullptr, "WebGpuContext::create must return valid context");
}

WindowedApplication::~WindowedApplication() {
    if (m_depth_view) {
        wgpuTextureViewRelease(m_depth_view);
    }
    if (m_depth_texture) {
        wgpuTextureDestroy(m_depth_texture);
        wgpuTextureRelease(m_depth_texture);
    }
}

void WindowedApplication::run() {
    init_windowing();
#if defined(__EMSCRIPTEN__)
    Application::run();
#else
    while (!m_viewport->should_close() && !should_stop()) {
        run_one_frame();
        check_frame_limit();
    }
#endif
}

bool WindowedApplication::ensure_webgpu_ready() {
    if (m_webgpu_context->is_failed()) {
        return false;
    }

    if (m_webgpu_context->is_initializing()) {
        m_webgpu_context->tick_init();

        if (m_webgpu_context->is_failed()) {
            log(pts::LogLevel::Error, "WebGPU context initialization failed");
            m_viewport->request_close();
            return false;
        }

        if (m_webgpu_context->is_initializing()) {
            return false;
        }

        log(pts::LogLevel::Info, "Application initialized");
    }

    return true;
}

void WindowedApplication::ensure_depth_buffer(uint32_t w, uint32_t h) {
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
    tex_desc.format = WGPUTextureFormat_Depth24Plus;
    tex_desc.usage = WGPUTextureUsage_RenderAttachment;
    tex_desc.mipLevelCount = 1;
    tex_desc.sampleCount = 1;
    tex_desc.dimension = WGPUTextureDimension_2D;
    m_depth_texture = wgpuDeviceCreateTexture(device.handle(), &tex_desc);

    WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    view_desc.format = WGPUTextureFormat_Depth24Plus;
    view_desc.dimension = WGPUTextureViewDimension_2D;
    view_desc.mipLevelCount = 1;
    view_desc.arrayLayerCount = 1;
    m_depth_view = wgpuTextureCreateView(m_depth_texture, &view_desc);

    m_depth_width = w;
    m_depth_height = h;
}

void WindowedApplication::run_one_frame() {
    m_windowing->pump_events(pts::rendering::PumpEventMode::Poll);

    if (!ensure_webgpu_ready()) {
        return;
    }

    if (!m_ready) {
        on_ready();
        m_ready = true;
    }

    if (m_framebuffer_resized) {
        auto const extent = m_viewport->drawable_extent();
        m_webgpu_context->surface().resize(extent);
        on_resize(extent.w, extent.h);
        m_framebuffer_resized = false;
    }

    update(get_delta_time());

    auto& surface = m_webgpu_context->surface();
    WGPUTextureView surface_view = surface.acquire_texture_view();
    if (!surface_view) {
        return;
    }

    auto const extent = m_viewport->drawable_extent();
    ensure_depth_buffer(extent.w, extent.h);

    auto const& device = m_webgpu_context->device();

    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device.handle(), &enc_desc);

    FrameContext ctx{device,       encoder,  surface_view, m_webgpu_context->surface_format(),
                     m_depth_view, extent.w, extent.h};
    render(ctx);

    WGPUCommandBufferDescriptor cmd_desc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(device.queue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    surface.present();
}

auto WindowedApplication::webgpu_context() noexcept -> rendering::WebGpuContext* {
    return m_webgpu_context.get();
}

auto WindowedApplication::webgpu_context() const noexcept -> const rendering::WebGpuContext* {
    return m_webgpu_context.get();
}

auto WindowedApplication::windowing() noexcept -> rendering::IWindowing* {
    return m_windowing.get();
}

auto WindowedApplication::viewport() noexcept -> rendering::IViewport* {
    return m_viewport.get();
}

auto WindowedApplication::viewport() const noexcept -> const rendering::IViewport* {
    return m_viewport.get();
}

auto WindowedApplication::window_width() const noexcept -> int {
    return static_cast<int>(m_viewport->drawable_extent().w);
}

auto WindowedApplication::window_height() const noexcept -> int {
    return static_cast<int>(m_viewport->drawable_extent().h);
}

auto WindowedApplication::depth_view() const noexcept -> WGPUTextureView {
    return m_depth_view;
}

}  // namespace pts
