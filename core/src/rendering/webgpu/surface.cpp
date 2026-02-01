#include <core/diagnostics.h>
#include <core/rendering/webgpu/device.h>
#include <core/rendering/webgpu/surface.h>
#include <core/scopeUtils.h>

#include <cstring>

namespace pts::webgpu {
namespace {
constexpr const char* k_default_canvas_selector = "#canvas";

auto choose_format(const WGPUSurfaceCapabilities& capabilities) -> WGPUTextureFormat {
    if (capabilities.formatCount == 0 || capabilities.formats == nullptr) {
        return WGPUTextureFormat_BGRA8Unorm;
    }
    for (size_t i = 0; i < capabilities.formatCount; ++i) {
        if (capabilities.formats[i] == WGPUTextureFormat_BGRA8Unorm) {
            return capabilities.formats[i];
        }
    }
    return capabilities.formats[0];
}

auto choose_present_mode(const WGPUSurfaceCapabilities& capabilities) -> WGPUPresentMode {
    if (capabilities.presentModeCount == 0 || capabilities.presentModes == nullptr) {
        return WGPUPresentMode_Fifo;
    }
    for (size_t i = 0; i < capabilities.presentModeCount; ++i) {
        if (capabilities.presentModes[i] == WGPUPresentMode_Fifo) {
            return capabilities.presentModes[i];
        }
    }
    return capabilities.presentModes[0];
}

auto choose_alpha_mode(const WGPUSurfaceCapabilities& capabilities) -> WGPUCompositeAlphaMode {
    if (capabilities.alphaModeCount == 0 || capabilities.alphaModes == nullptr) {
        return WGPUCompositeAlphaMode_Auto;
    }
    for (size_t i = 0; i < capabilities.alphaModeCount; ++i) {
        if (capabilities.alphaModes[i] == WGPUCompositeAlphaMode_Auto) {
            return capabilities.alphaModes[i];
        }
    }
    return capabilities.alphaModes[0];
}

auto create_surface_for_handle(WGPUInstance instance,
                               const rendering::NativeViewportHandle& handle) -> WGPUSurface {
    if (instance == nullptr) {
        return nullptr;
    }

    WGPUSurfaceDescriptor descriptor = WGPU_SURFACE_DESCRIPTOR_INIT;

    switch (handle.platform) {
        case rendering::NativePlatform::win32: {
            WGPUSurfaceSourceWindowsHWND source = WGPU_SURFACE_SOURCE_WINDOWS_HWND_INIT;
            source.hinstance = handle.win32.hinstance;
            source.hwnd = handle.win32.hwnd;
            descriptor.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&source);
            return wgpuInstanceCreateSurface(instance, &descriptor);
        }
        case rendering::NativePlatform::xlib: {
            WGPUSurfaceSourceXlibWindow source = WGPU_SURFACE_SOURCE_XLIB_WINDOW_INIT;
            source.display = handle.xlib.display;
            source.window = handle.xlib.window;
            descriptor.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&source);
            return wgpuInstanceCreateSurface(instance, &descriptor);
        }
        case rendering::NativePlatform::emscripten: {
#ifdef __EMSCRIPTEN__
            const char* selector =
                handle.web.canvas_selector ? handle.web.canvas_selector : k_default_canvas_selector;
            WGPUEmscriptenSurfaceSourceCanvasHTMLSelector source =
                WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
            source.selector = WGPUStringView{selector, std::strlen(selector)};
            descriptor.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&source);
            return wgpuInstanceCreateSurface(instance, &descriptor);
#else
            return nullptr;
#endif
        }
        default:
            return nullptr;
    }
}
}  // namespace

Surface::Surface(WGPUSurface surface, WGPUDevice device, WGPUTextureFormat format,
                 WGPUTextureUsage usage, WGPUPresentMode present_mode,
                 WGPUCompositeAlphaMode alpha_mode, rendering::Extent2D extent)
    : m_surface(surface),
      m_device(device),
      m_format(format),
      m_usage(usage),
      m_present_mode(present_mode),
      m_alpha_mode(alpha_mode) {
    // Enforce class invariants: surface and device must be valid
    INVARIANT_MSG(m_surface != nullptr, "Surface handle must be valid");
    INVARIANT_MSG(m_device != nullptr, "Device handle must be valid");
    configure(extent.w, extent.h);
}

Surface::Surface(Surface&& other) noexcept
    : m_surface(other.m_surface),
      m_device(other.m_device),
      m_format(other.m_format),
      m_usage(other.m_usage),
      m_present_mode(other.m_present_mode),
      m_alpha_mode(other.m_alpha_mode),
      m_current_texture(other.m_current_texture),
      m_current_view(other.m_current_view),
      m_present_pending(other.m_present_pending),
      m_width(other.m_width),
      m_height(other.m_height) {
    other.m_surface = nullptr;
    other.m_device = nullptr;
    other.m_current_texture = nullptr;
    other.m_current_view = nullptr;
    other.m_present_pending = false;
    other.m_width = 0;
    other.m_height = 0;
}

auto Surface::operator=(Surface&& other) noexcept -> Surface& {
    if (this != &other) {
        if (m_current_view != nullptr) {
            wgpuTextureViewRelease(m_current_view);
        }
        if (m_current_texture != nullptr) {
            wgpuTextureRelease(m_current_texture);
        }
        if (m_surface != nullptr) {
            wgpuSurfaceRelease(m_surface);
        }

        m_surface = other.m_surface;
        m_device = other.m_device;
        m_format = other.m_format;
        m_usage = other.m_usage;
        m_present_mode = other.m_present_mode;
        m_alpha_mode = other.m_alpha_mode;
        m_current_texture = other.m_current_texture;
        m_current_view = other.m_current_view;
        m_present_pending = other.m_present_pending;
        m_width = other.m_width;
        m_height = other.m_height;

        other.m_surface = nullptr;
        other.m_device = nullptr;
        other.m_current_texture = nullptr;
        other.m_current_view = nullptr;
        other.m_present_pending = false;
        other.m_width = 0;
        other.m_height = 0;
    }
    return *this;
}

Surface::~Surface() {
    if (m_current_view != nullptr) {
        wgpuTextureViewRelease(m_current_view);
    }
    if (m_current_texture != nullptr) {
        wgpuTextureRelease(m_current_texture);
    }
    if (m_surface != nullptr) {
        wgpuSurfaceRelease(m_surface);
    }
}

auto Surface::create(const Device& device, const rendering::NativeViewportHandle& handle,
                     rendering::Extent2D extent) -> Surface {
    // Device is always valid due to enforced invariants

    WGPUSurface surface = create_surface_for_handle(device.instance(), handle);
    if (surface == nullptr) {
        throw std::runtime_error("Failed to create WebGPU surface for viewport");
    }

    // Ensure surface is released if construction fails
    SCOPE_FAIL {
        if (surface) wgpuSurfaceRelease(surface);
    };

    WGPUTextureFormat format = WGPUTextureFormat_BGRA8Unorm;
    WGPUPresentMode present_mode = WGPUPresentMode_Fifo;
    WGPUCompositeAlphaMode alpha_mode = WGPUCompositeAlphaMode_Auto;
    WGPUTextureUsage usage = WGPUTextureUsage_RenderAttachment;

    WGPUAdapter adapter = wgpuDeviceGetAdapter(device.handle());
    SCOPE_EXIT {
        if (adapter) wgpuAdapterRelease(adapter);
    };

    if (adapter != nullptr) {
        WGPUSurfaceCapabilities capabilities = WGPU_SURFACE_CAPABILITIES_INIT;
        if (wgpuSurfaceGetCapabilities(surface, adapter, &capabilities) == WGPUStatus_Success) {
            // Fail fast if RenderAttachment usage is not supported
            if ((capabilities.usages & WGPUTextureUsage_RenderAttachment) == 0) {
                wgpuSurfaceCapabilitiesFreeMembers(capabilities);
                throw std::runtime_error(
                    "Surface does not support WGPUTextureUsage_RenderAttachment");
            }

            format = choose_format(capabilities);
            present_mode = choose_present_mode(capabilities);
            alpha_mode = choose_alpha_mode(capabilities);
            wgpuSurfaceCapabilitiesFreeMembers(capabilities);
        }
    }

    return Surface(surface, device.handle(), format, usage, present_mode, alpha_mode, extent);
}

auto Surface::format() const noexcept -> WGPUTextureFormat {
    return m_format;
}

auto Surface::extent() const noexcept -> rendering::Extent2D {
    return {m_width, m_height};
}

void Surface::resize(rendering::Extent2D extent) {
    // Surface is always valid due to enforced invariants
    if (extent.w == 0 || extent.h == 0) {
        m_width = 0;
        m_height = 0;
        return;
    }
    if (extent.w == m_width && extent.h == m_height) {
        return;
    }

    // Release any in-flight surface resources before reconfiguring
    if (m_current_view != nullptr) {
        wgpuTextureViewRelease(m_current_view);
        m_current_view = nullptr;
    }
    if (m_current_texture != nullptr) {
        wgpuTextureRelease(m_current_texture);
        m_current_texture = nullptr;
    }
    m_present_pending = false;

    configure(extent.w, extent.h);
}

auto Surface::acquire_texture_view() -> WGPUTextureView {
    // Surface is always valid due to enforced invariants
    if (m_width == 0 || m_height == 0) {
        return nullptr;
    }

    if (m_current_view != nullptr) {
        wgpuTextureViewRelease(m_current_view);
        m_current_view = nullptr;
    }
    if (m_current_texture != nullptr) {
        wgpuTextureRelease(m_current_texture);
        m_current_texture = nullptr;
    }
    m_present_pending = false;

    WGPUSurfaceTexture surface_texture = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(m_surface, &surface_texture);

    // Ensure surface texture is released if not successfully assigned to m_current_texture
    SCOPE_FAIL {
        if (surface_texture.texture) wgpuTextureRelease(surface_texture.texture);
    };

    if (surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        if (surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Outdated) {
            configure(m_width, m_height);
        }
        return nullptr;
    }

    m_current_texture = surface_texture.texture;
    if (m_current_texture == nullptr) {
        return nullptr;
    }

    m_current_view = wgpuTextureCreateView(m_current_texture, nullptr);
    if (m_current_view == nullptr) {
        wgpuTextureRelease(m_current_texture);
        m_current_texture = nullptr;
        return nullptr;
    }
    m_present_pending = true;
    return m_current_view;
}

void Surface::present() {
    // Surface is always valid due to enforced invariants
    if (!m_present_pending || m_surface == nullptr) {
        return;
    }
    wgpuSurfacePresent(m_surface);
    if (m_current_view != nullptr) {
        wgpuTextureViewRelease(m_current_view);
        m_current_view = nullptr;
    }
    if (m_current_texture != nullptr) {
        wgpuTextureRelease(m_current_texture);
        m_current_texture = nullptr;
    }
    m_present_pending = false;
}

void Surface::configure(uint32_t width, uint32_t height) {
    // Surface is always valid due to enforced invariants
    if (width == 0 || height == 0) {
        return;
    }
    WGPUSurfaceConfiguration configuration = WGPU_SURFACE_CONFIGURATION_INIT;
    configuration.device = m_device;
    configuration.format = m_format;
    configuration.usage = m_usage;
    configuration.width = width;
    configuration.height = height;
    configuration.presentMode = m_present_mode;
    configuration.alphaMode = m_alpha_mode;
    wgpuSurfaceConfigure(m_surface, &configuration);
    m_width = width;
    m_height = height;
}

}  // namespace pts::webgpu
