#include <core/diagnostics.h>
#include <core/loggingManager.h>
#include <core/rendering/webgpu/errorScope.h>
#include <core/rendering/webgpuContext.h>
#include <core/scopeUtils.h>

namespace {
constexpr const char* k_webgpu_logger_name = "webgpu";
}

namespace pts::rendering {

WebGpuContext::WebGpuContext(PrivateCtorTag, std::shared_ptr<spdlog::logger> logger)
    : m_state(FailedState{}), m_logger(std::move(logger)) {
    INVARIANT_MSG(m_logger != nullptr, "logger is null");
}

WebGpuContext::~WebGpuContext() {
    release_resources();
}

WebGpuContext::WebGpuContext(WebGpuContext&& other) noexcept
    : m_state(std::move(other.m_state)), m_logger(std::move(other.m_logger)) {
    // Moving during initialization is unsafe: async callbacks hold raw pointers to
    // InitializingState
    INVARIANT_MSG(!std::holds_alternative<InitializingState>(m_state),
                  "WebGpuContext cannot be moved during initialization");
    other.m_state = FailedState{};
}

auto WebGpuContext::operator=(WebGpuContext&& other) noexcept -> WebGpuContext& {
    if (this != &other) {
        // Moving during initialization is unsafe: async callbacks hold raw pointers to
        // InitializingState
        INVARIANT_MSG(!std::holds_alternative<InitializingState>(m_state),
                      "WebGpuContext cannot be move-assigned to during initialization");
        INVARIANT_MSG(!std::holds_alternative<InitializingState>(other.m_state),
                      "WebGpuContext cannot be move-assigned from during initialization");
        release_resources();
        m_state = std::move(other.m_state);
        m_logger = std::move(other.m_logger);
        other.m_state = FailedState{};
    }
    return *this;
}

auto WebGpuContext::create(const IViewport& viewport,
                           pts::LoggingManager& logging_manager) -> std::unique_ptr<WebGpuContext> {
    auto logger = logging_manager.get_logger_shared(k_webgpu_logger_name);

    auto context = std::make_unique<WebGpuContext>(PrivateCtorTag{}, logger);

    // Start in Initializing state
    InitializingState init_state{};
    init_state.viewport_handle = viewport.native_handle();
    init_state.viewport_extent = viewport.drawable_extent();

    // Create WebGPU instance
    WGPUInstanceDescriptor instance_descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    init_state.instance = wgpuCreateInstance(&instance_descriptor);
    if (init_state.instance == nullptr) {
        logger->error("Failed to create WebGPU instance");
        context->m_state = FailedState{};
        return context;
    }

    logger->debug("Using Dawn WebGPU backend");

    context->m_state = std::move(init_state);
    logger->info("Starting WebGPU context initialization...");
    context->start_adapter_request();

    return context;
}

void WebGpuContext::tick_init() {
    auto* init_state = std::get_if<InitializingState>(&m_state);
    if (init_state == nullptr) {
        return;
    }

    // Process WebGPU events to advance callbacks
    wgpuInstanceProcessEvents(init_state->instance);

    switch (init_state->phase) {
        case InitPhase::RequestingAdapter:
            if (init_state->adapter_request_done) {
                if (init_state->adapter_status != WGPURequestAdapterStatus_Success ||
                    init_state->adapter == nullptr) {
                    m_logger->error("Failed to request WebGPU adapter (status: {})",
                                    static_cast<int>(init_state->adapter_status));
                    set_failed();
                    return;
                }
                m_logger->debug("WebGPU adapter acquired successfully");
                start_device_request();
            }
            break;

        case InitPhase::RequestingDevice:
            if (init_state->device_request_done) {
                if (init_state->device_status != WGPURequestDeviceStatus_Success ||
                    init_state->device == nullptr) {
                    m_logger->error("Failed to request WebGPU device (status: {})",
                                    static_cast<int>(init_state->device_status));
                    set_failed();
                    return;
                }
                m_logger->debug("WebGPU device acquired successfully");
                init_state->phase = InitPhase::CreatingSurface;
                finish_initialization();
            }
            break;

        case InitPhase::CreatingSurface:
            break;
    }
}

void WebGpuContext::start_adapter_request() {
    auto* init_state = std::get_if<InitializingState>(&m_state);
    PRECONDITION(init_state != nullptr);
    PRECONDITION(init_state->instance != nullptr);

    WGPURequestAdapterOptions options = {};
    options.backendType = WGPUBackendType_Undefined;

    WGPURequestAdapterCallbackInfo callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_AllowProcessEvents;
    callback.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView,
                           void* userdata1, void*) {
        PRECONDITION(userdata1 != nullptr);
        auto* init_data = static_cast<InitializingState*>(userdata1);
        init_data->adapter_status = status;
        init_data->adapter = adapter;
        init_data->adapter_request_done = true;
    };
    callback.userdata1 = init_state;

    m_logger->debug("Requesting WebGPU adapter...");
    wgpuInstanceRequestAdapter(init_state->instance, &options, callback);
    init_state->phase = InitPhase::RequestingAdapter;
}

void WebGpuContext::start_device_request() {
    auto* init_state = std::get_if<InitializingState>(&m_state);
    PRECONDITION(init_state != nullptr);
    PRECONDITION(init_state->adapter != nullptr);

    // Setup device descriptor with error callbacks
    WGPUDeviceDescriptor device_descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;

    // Configure device lost callback
    WGPUDeviceLostCallbackInfo device_lost_callback = WGPU_DEVICE_LOST_CALLBACK_INFO_INIT;
    device_lost_callback.mode = WGPUCallbackMode_AllowSpontaneous;
    device_lost_callback.callback = [](WGPUDevice const*, WGPUDeviceLostReason reason,
                                       WGPUStringView message, void*, void*) {
        const char* reason_str = "Unknown";
        switch (reason) {
            case WGPUDeviceLostReason_Unknown:
                reason_str = "Unknown";
                break;
            case WGPUDeviceLostReason_Destroyed:
                reason_str = "Destroyed";
                break;
            case WGPUDeviceLostReason_CallbackCancelled:
                reason_str = "CallbackCancelled";
                break;
            case WGPUDeviceLostReason_FailedCreation:
                reason_str = "FailedCreation";
                break;
            default:
                break;
        }
        auto msg = message.data ? std::string_view(message.data, message.length)
                                : std::string_view("(no message)");
        // Use stderr since logger may not be available
        std::fprintf(stderr, "[WebGPU Device Lost] Reason: %s, Message: %.*s\n", reason_str,
                     static_cast<int>(msg.size()), msg.data());
    };
    device_descriptor.deviceLostCallbackInfo = device_lost_callback;

    // Configure uncaptured error callback
    WGPUUncapturedErrorCallbackInfo uncaptured_error_callback =
        WGPU_UNCAPTURED_ERROR_CALLBACK_INFO_INIT;
    uncaptured_error_callback.callback = [](WGPUDevice const*, WGPUErrorType type,
                                            WGPUStringView message, void*, void*) {
        if (type == WGPUErrorType_NoError) {
            return;
        }
        const char* type_str = pts::webgpu::error_type_name(type);
        auto msg = message.data ? std::string_view(message.data, message.length)
                                : std::string_view("(no message)");
        std::fprintf(stderr, "[WebGPU Uncaptured Error] Type: %s, Message: %.*s\n", type_str,
                     static_cast<int>(msg.size()), msg.data());
    };
    device_descriptor.uncapturedErrorCallbackInfo = uncaptured_error_callback;

    WGPURequestDeviceCallbackInfo callback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_AllowProcessEvents;
    callback.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView,
                           void* userdata1, void*) {
        PRECONDITION(userdata1 != nullptr);
        auto* init_data = static_cast<InitializingState*>(userdata1);
        init_data->device_status = status;
        init_data->device = device;
        init_data->device_request_done = true;
    };
    callback.userdata1 = init_state;

    m_logger->debug("Requesting WebGPU device...");
    wgpuAdapterRequestDevice(init_state->adapter, &device_descriptor, callback);
    init_state->phase = InitPhase::RequestingDevice;
}

void WebGpuContext::finish_initialization() {
    auto* init_state = std::get_if<InitializingState>(&m_state);
    PRECONDITION(init_state != nullptr);
    PRECONDITION(init_state->device != nullptr);

    // Get queue
    WGPUQueue queue = wgpuDeviceGetQueue(init_state->device);
    if (queue == nullptr) {
        m_logger->error("Failed to get WebGPU queue");
        set_failed();
        return;
    }
    init_state->queue = queue;

    m_logger->debug("WebGPU queue acquired successfully");

    // Create Device wrapper (takes ownership of handles)
    pts::webgpu::Device device_wrapper(init_state->instance, init_state->device, init_state->queue,
                                       m_logger);
    // Device now owns these handles, clear from init_state to prevent double-release
    init_state->instance = nullptr;
    init_state->device = nullptr;
    init_state->queue = nullptr;

    // Create Surface
    pts::webgpu::Surface surface_wrapper = pts::webgpu::Surface::create(
        device_wrapper, init_state->viewport_handle, init_state->viewport_extent);

    // Release adapter (no longer needed)
    if (init_state->adapter) {
        wgpuAdapterRelease(init_state->adapter);
        init_state->adapter = nullptr;
    }

    // Transition to Ready state
    m_state = ReadyState(std::move(device_wrapper), std::move(surface_wrapper));

    m_logger->info("WebGPU context created successfully (Dawn backend)");
}

void WebGpuContext::set_failed() {
    release_resources();
    m_state = FailedState{};
}

void WebGpuContext::release_resources() {
    std::visit(
        [](auto& state) {
            using T = std::decay_t<decltype(state)>;
            if constexpr (std::is_same_v<T, InitializingState>) {
                if (state.queue) {
                    wgpuQueueRelease(state.queue);
                }
                if (state.device) {
                    wgpuDeviceRelease(state.device);
                }
                if (state.adapter) {
                    wgpuAdapterRelease(state.adapter);
                }
                if (state.instance) {
                    wgpuInstanceRelease(state.instance);
                }
            }
            // ReadyState cleanup is handled by Device and Surface destructors
        },
        m_state);
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
