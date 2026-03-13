#include <core/diagnostics.h>
#include <core/loggingManager.h>
#include <core/rendering/webgpuContext.h>

namespace {
constexpr const char* k_webgpu_logger_name = "WebGPU";
}

namespace pts::rendering {

WebGpuContext::WebGpuContext(PrivateCtorTag, std::shared_ptr<spdlog::logger> logger)
    : m_logger(std::move(logger)) {
    INVARIANT_MSG(m_logger != nullptr, "logger is null");
}

WebGpuContext::~WebGpuContext() = default;

WebGpuContext::WebGpuContext(WebGpuContext&& other) noexcept
    : Base(std::move(other)), m_logger(std::move(other.m_logger)) {
}

auto WebGpuContext::operator=(WebGpuContext&& other) noexcept -> WebGpuContext& {
    if (this != &other) {
        Base::operator=(std::move(other));
        m_logger = std::move(other.m_logger);
    }
    return *this;
}

auto WebGpuContext::create(const IViewport& viewport, pts::LoggingManager& logging_manager)
    -> std::unique_ptr<WebGpuContext> {
    auto logger = logging_manager.get_logger_shared(k_webgpu_logger_name);

    auto context = std::make_unique<WebGpuContext>(PrivateCtorTag{}, logger);

    auto device = pts::webgpu::Device::create_async(logger);
    if (device->is<pts::webgpu::DeviceFailedState>()) {
        context->transition<ContextFailedState>();
        return context;
    }

    ContextInitializingState init_state{};
    init_state.viewport_handle = viewport.native_handle();
    init_state.viewport_extent = viewport.drawable_extent();
    init_state.device = std::move(device);

    context->transition<ContextInitializingState>(std::move(init_state));
    logger->info("Starting WebGPU context initialization...");

    return context;
}

void WebGpuContext::on_tick() {
    auto* init_state = get_if<ContextInitializingState>();
    if (init_state == nullptr) {
        return;
    }

    PRECONDITION(init_state->device != nullptr);

    // Advance device initialization
    init_state->device->tick();

    if (init_state->device->is<pts::webgpu::DeviceFailedState>()) {
        set_failed();
        return;
    }

    if (init_state->device->is<pts::webgpu::DeviceReadyState>()) {
        finish_initialization();
    }
}

auto WebGpuContext::is_pending() const -> bool {
    return is<ContextInitializingState>();
}

auto WebGpuContext::wgpu_instance() const -> WGPUInstance {
    // Device handles its own event processing in its tick_init(), so we return
    // nullptr here. The base's tick() skips wgpuInstanceProcessEvents when null.
    return nullptr;
}

void WebGpuContext::finish_initialization() {
    auto* init_state = get_if<ContextInitializingState>();
    PRECONDITION(init_state != nullptr);
    PRECONDITION(init_state->device != nullptr);
    PRECONDITION(init_state->device->is<pts::webgpu::DeviceReadyState>());

    // Create Surface
    pts::webgpu::Surface surface_wrapper = pts::webgpu::Surface::create(
        *init_state->device, init_state->viewport_handle, init_state->viewport_extent);

    // Transition to Ready state
    transition<ContextReadyState>(std::move(*init_state->device), std::move(surface_wrapper));

    m_logger->info("WebGPU context created successfully");
}

void WebGpuContext::set_failed() {
    transition<ContextFailedState>();
}

auto WebGpuContext::device() const noexcept -> const pts::webgpu::Device& {
    PRECONDITION_MSG(is<ContextReadyState>(), "device() called when not Ready");
    return get<ContextReadyState>().device;
}

auto WebGpuContext::surface() noexcept -> pts::webgpu::Surface& {
    PRECONDITION_MSG(is<ContextReadyState>(), "surface() called when not Ready");
    return get<ContextReadyState>().surface;
}

auto WebGpuContext::surface_format() const noexcept -> WGPUTextureFormat {
    PRECONDITION_MSG(is<ContextReadyState>(), "surface_format() called when not Ready");
    return get<ContextReadyState>().surface.format();
}

}  // namespace pts::rendering
