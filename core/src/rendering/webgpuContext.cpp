#include <core/diagnostics.h>
#include <core/loggingManager.h>
#include <core/rendering/webgpuContext.h>

namespace {
constexpr const char* k_webgpu_logger_name = "WebGPU";
}

namespace pts::rendering {

WebGpuContext::WebGpuContext(PrivateCtorTag, std::shared_ptr<spdlog::logger> logger)
    : m_state(FailedState{}), m_logger(std::move(logger)) {
    INVARIANT_MSG(m_logger != nullptr, "logger is null");
}

WebGpuContext::~WebGpuContext() = default;

WebGpuContext::WebGpuContext(WebGpuContext&& other) noexcept
    : m_state(std::move(other.m_state)), m_logger(std::move(other.m_logger)) {
    other.m_state = FailedState{};
}

auto WebGpuContext::operator=(WebGpuContext&& other) noexcept -> WebGpuContext& {
    if (this != &other) {
        m_state = std::move(other.m_state);
        m_logger = std::move(other.m_logger);
        other.m_state = FailedState{};
    }
    return *this;
}

auto WebGpuContext::create(const IViewport& viewport, pts::LoggingManager& logging_manager)
    -> std::unique_ptr<WebGpuContext> {
    auto logger = logging_manager.get_logger_shared(k_webgpu_logger_name);

    auto context = std::make_unique<WebGpuContext>(PrivateCtorTag{}, logger);

    auto device = pts::webgpu::Device::create_async(logger);
    if (device->is_failed()) {
        context->m_state = FailedState{};
        return context;
    }

    InitializingState init_state{};
    init_state.viewport_handle = viewport.native_handle();
    init_state.viewport_extent = viewport.drawable_extent();
    init_state.device = std::move(device);

    context->m_state = std::move(init_state);
    logger->info("Starting WebGPU context initialization...");

    return context;
}

void WebGpuContext::tick_init() {
    auto* init_state = std::get_if<InitializingState>(&m_state);
    if (init_state == nullptr) {
        return;
    }

    PRECONDITION(init_state->device != nullptr);

    // Advance device initialization
    init_state->device->tick_init();

    if (init_state->device->is_failed()) {
        set_failed();
        return;
    }

    if (init_state->device->is_ready()) {
        finish_initialization();
    }
}

void WebGpuContext::finish_initialization() {
    auto* init_state = std::get_if<InitializingState>(&m_state);
    PRECONDITION(init_state != nullptr);
    PRECONDITION(init_state->device != nullptr);
    PRECONDITION(init_state->device->is_ready());

    // Create Surface
    pts::webgpu::Surface surface_wrapper = pts::webgpu::Surface::create(
        *init_state->device, init_state->viewport_handle, init_state->viewport_extent);

    // Transition to Ready state
    m_state = ReadyState(std::move(*init_state->device), std::move(surface_wrapper));

    m_logger->info("WebGPU context created successfully");
}

void WebGpuContext::set_failed() {
    m_state = FailedState{};
}

auto WebGpuContext::get_state_enum() const noexcept -> WebGpuContextState {
    return std::visit(
        [](const auto& state) -> WebGpuContextState {
            using T = std::decay_t<decltype(state)>;
            if constexpr (std::is_same_v<T, InitializingState>) {
                return WebGpuContextState::Initializing;
            } else if constexpr (std::is_same_v<T, ReadyState>) {
                return WebGpuContextState::Ready;
            } else {
                return WebGpuContextState::Failed;
            }
        },
        m_state);
}

auto WebGpuContext::state() const noexcept -> WebGpuContextState {
    return get_state_enum();
}

auto WebGpuContext::is_ready() const noexcept -> bool {
    return std::holds_alternative<ReadyState>(m_state);
}

auto WebGpuContext::is_failed() const noexcept -> bool {
    return std::holds_alternative<FailedState>(m_state);
}

auto WebGpuContext::is_initializing() const noexcept -> bool {
    return std::holds_alternative<InitializingState>(m_state);
}

auto WebGpuContext::device() const noexcept -> const pts::webgpu::Device& {
    PRECONDITION_MSG(is_ready(), "device() called when not Ready");
    return std::get<ReadyState>(m_state).device;
}

auto WebGpuContext::surface() noexcept -> pts::webgpu::Surface& {
    PRECONDITION_MSG(is_ready(), "surface() called when not Ready");
    return std::get<ReadyState>(m_state).surface;
}

auto WebGpuContext::surface_format() const noexcept -> WGPUTextureFormat {
    PRECONDITION_MSG(is_ready(), "surface_format() called when not Ready");
    return std::get<ReadyState>(m_state).surface.format();
}

}  // namespace pts::rendering
