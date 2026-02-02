#include <core/diagnostics.h>
#include <core/loggingManager.h>
#include <core/rendering/webgpu/device.h>
#include <core/rendering/webgpu/errorScope.h>
#include <core/scopeUtils.h>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

#include "logging.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/html5_webgpu.h>
#endif

namespace pts::webgpu {

Device::Device(WGPUInstance instance, WGPUDevice device, WGPUQueue queue,
               std::shared_ptr<spdlog::logger> logger)
    : m_logger(std::move(logger)) {
    INVARIANT_MSG(instance != nullptr, "instance handle is null");
    INVARIANT_MSG(device != nullptr, "device handle is null");
    INVARIANT_MSG(queue != nullptr, "queue handle is null");
    INVARIANT_MSG(m_logger != nullptr, "logger is null");

    m_state = ReadyState{instance, device, queue};

    m_logger->debug("Device constructed successfully (device={}, queue={})",
                    static_cast<void*>(device), static_cast<void*>(queue));
}

Device::Device(PrivateCtorTag, std::shared_ptr<spdlog::logger> logger, InitializingState init_state)
    : m_state(std::move(init_state)), m_logger(std::move(logger)) {
    INVARIANT_MSG(m_logger != nullptr, "logger is null");
}

Device::Device(Device&& other) noexcept
    : m_state(std::move(other.m_state)), m_logger(std::move(other.m_logger)) {
    other.m_state = FailedState{};
}

auto Device::operator=(Device&& other) noexcept -> Device& {
    if (this != &other) {
        release_resources();
        m_state = std::move(other.m_state);
        m_logger = std::move(other.m_logger);
        other.m_state = FailedState{};
    }
    return *this;
}

Device::~Device() {
    release_resources();
}

auto Device::create(std::shared_ptr<spdlog::logger> logger) -> Device {
    PRECONDITION_MSG(logger != nullptr, "logger is null");

    auto device = create_async(logger);
    POSTCONDITION_MSG(device != nullptr, "create_async returned nullptr");

    // Poll until ready or failed
    while (!device->is_ready() && !device->is_failed()) {
        device->tick_init();
        std::this_thread::yield();
    }

    if (device->is_failed()) {
        throw std::runtime_error("Failed to create WebGPU device");
    }

    return std::move(*device);
}

auto Device::create_async(std::shared_ptr<spdlog::logger> logger) -> std::unique_ptr<Device> {
    PRECONDITION_MSG(logger != nullptr, "logger is null");

    logger->info("Creating WebGPU device...");

    InitializingState init_state{};

    // Create WebGPU instance
    WGPUInstanceDescriptor instance_descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    init_state.instance = wgpuCreateInstance(&instance_descriptor);
    if (init_state.instance == nullptr) {
        logger->error("Failed to create WebGPU instance");
        auto device = std::make_unique<Device>(PrivateCtorTag{}, logger, InitializingState{});
        device->set_failed();
        return device;
    }

#ifdef __EMSCRIPTEN__
    logger->debug("Using Emscripten/Browser WebGPU backend");
#else
    logger->debug("Using Dawn/Native WebGPU backend");
#endif

    auto device = std::make_unique<Device>(PrivateCtorTag{}, logger, std::move(init_state));
    device->start_adapter_request();

    return device;
}

void Device::tick_init() {
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
                finish_initialization();
            }
            break;
    }
}

void Device::start_adapter_request() {
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

void Device::start_device_request() {
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
        pts::LogLevel level = pts::LogLevel::Error;
        switch (reason) {
            case WGPUDeviceLostReason_Unknown:
                reason_str = "Unknown";
                break;
            case WGPUDeviceLostReason_Destroyed:
                reason_str = "Destroyed";
                level = pts::LogLevel::Info;
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
        pts::log_or_cerr(k_webgpu_logger_name, level,
                         "[WebGPU Device Lost] Reason: {}, Message: {}", reason_str,
                         message.data ? std::string_view(message.data, message.length)
                                      : std::string_view("(no message)"));
    };
    device_descriptor.deviceLostCallbackInfo = device_lost_callback;

    // Configure uncaptured error callback
    WGPUUncapturedErrorCallbackInfo uncaptured_error_callback =
        WGPU_UNCAPTURED_ERROR_CALLBACK_INFO_INIT;
    uncaptured_error_callback.callback = [](WGPUDevice const*, WGPUErrorType type,
                                            WGPUStringView message, void*, void*) {
        if (type == WGPUErrorType_NoError) {
            return;  // Don't log "no error"
        }
        pts::log_or_cerr(k_webgpu_logger_name, pts::LogLevel::Error,
                         "[WebGPU Uncaptured Error] Type: {}, Message: {}",
                         pts::webgpu::error_type_name(type),
                         message.data ? std::string_view(message.data, message.length)
                                      : std::string_view("(no message)"));
    };
    device_descriptor.uncapturedErrorCallbackInfo = uncaptured_error_callback;

    // Store InitializingState pointer in userdata for the callback to set device directly
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

void Device::finish_initialization() {
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
    m_logger->debug("WebGPU queue acquired successfully");

    // Release adapter (no longer needed)
    if (init_state->adapter) {
        wgpuAdapterRelease(init_state->adapter);
        init_state->adapter = nullptr;
    }

    // Transition to Ready state - move handles from InitializingState to ReadyState
    WGPUInstance instance = init_state->instance;
    WGPUDevice device = init_state->device;

    // Clear init_state handles before replacing state to prevent double-release
    init_state->instance = nullptr;
    init_state->device = nullptr;

    m_state = ReadyState{instance, device, queue};

#ifdef __EMSCRIPTEN__
    m_logger->info("WebGPU device created successfully (Emscripten/Browser backend)");
#else
    m_logger->info("WebGPU device created successfully (Dawn/Native backend)");
#endif
    m_logger->debug("Device lost and uncaptured error callbacks are registered");
}

void Device::set_failed() {
    release_resources();
    m_state = FailedState{};
}

void Device::release_resources() {
    std::visit(
        [](auto& state) {
            using T = std::decay_t<decltype(state)>;
            if constexpr (std::is_same_v<T, InitializingState>) {
                if (state.adapter) {
                    wgpuAdapterRelease(state.adapter);
                }
                if (state.device) {
                    wgpuDeviceRelease(state.device);
                }
                if (state.instance) {
                    wgpuInstanceRelease(state.instance);
                }
            } else if constexpr (std::is_same_v<T, ReadyState>) {
                if (state.queue) {
                    wgpuQueueRelease(state.queue);
                }
                if (state.device) {
                    wgpuDeviceRelease(state.device);
                }
                if (state.instance) {
                    wgpuInstanceRelease(state.instance);
                }
            }
        },
        m_state);
}

auto Device::get_state_enum() const noexcept -> DeviceState {
    return std::visit(
        [](const auto& state) -> DeviceState {
            using T = std::decay_t<decltype(state)>;
            if constexpr (std::is_same_v<T, InitializingState>) {
                return DeviceState::Initializing;
            } else if constexpr (std::is_same_v<T, ReadyState>) {
                return DeviceState::Ready;
            } else {
                return DeviceState::Failed;
            }
        },
        m_state);
}

auto Device::state() const noexcept -> DeviceState {
    return get_state_enum();
}

auto Device::is_ready() const noexcept -> bool {
    return std::holds_alternative<ReadyState>(m_state);
}

auto Device::is_failed() const noexcept -> bool {
    return std::holds_alternative<FailedState>(m_state);
}

auto Device::instance() const noexcept -> WGPUInstance {
    PRECONDITION_MSG(is_ready(), "instance() called when not Ready");
    return std::get<ReadyState>(m_state).instance;
}

auto Device::handle() const noexcept -> WGPUDevice {
    PRECONDITION_MSG(is_ready(), "handle() called when not Ready");
    return std::get<ReadyState>(m_state).device;
}

auto Device::queue() const noexcept -> WGPUQueue {
    PRECONDITION_MSG(is_ready(), "queue() called when not Ready");
    return std::get<ReadyState>(m_state).queue;
}

auto Device::create_buffer(std::size_t size, WGPUBufferUsage usage) const -> Buffer {
    PRECONDITION_MSG(is_ready(), "create_buffer() called when not Ready");
    const auto& ready = std::get<ReadyState>(m_state);
    m_logger->debug("Creating buffer (size={}, usage={})", size, usage);

    WGPUBufferDescriptor descriptor = {};
    descriptor.size = static_cast<uint64_t>(size);
    descriptor.usage = usage;
    descriptor.mappedAtCreation = false;

    ErrorScope error_scope(
        *this, {WGPUErrorFilter_Validation, WGPUErrorFilter_OutOfMemory, WGPUErrorFilter_Internal},
        m_logger->name(), "buffer");
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(ready.device, &descriptor);

    SCOPE_FAIL {
        m_logger->error("Failed to create WebGPU buffer (size={}, usage={})", size, usage);
        if (buffer != nullptr) {
            wgpuBufferRelease(buffer);
        }
    };

    error_scope.pop_and_throw_if_error();

    POSTCONDITION_MSG(buffer != nullptr, "wgpuDeviceCreateBuffer returned nullptr");
    m_logger->debug("Buffer created successfully (handle={})", static_cast<void*>(buffer));
    return Buffer(buffer, size);
}

auto Device::create_shader_module_from_source(std::string_view wgsl_source) const -> ShaderModule {
    PRECONDITION_MSG(is_ready(), "create_shader_module_from_source() called when not Ready");
    const auto& ready = std::get<ReadyState>(m_state);
    m_logger->debug("Creating shader module from source ({} bytes)", wgsl_source.size());

    if (wgsl_source.empty()) {
        throw std::invalid_argument("wgsl_source cannot be empty");
    }

    WGPUShaderSourceWGSL wgsl_descriptor = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl_descriptor.code = WGPUStringView{wgsl_source.data(), wgsl_source.size()};

    WGPUShaderModuleDescriptor descriptor = {};
    descriptor.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl_descriptor);

    ErrorScope error_scope(
        *this, {WGPUErrorFilter_Validation, WGPUErrorFilter_OutOfMemory, WGPUErrorFilter_Internal},
        m_logger->name(), "shader module");
    WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(ready.device, &descriptor);
    SCOPE_FAIL {
        m_logger->error("Failed to create shader module from source");
        if (shader_module != nullptr) {
            wgpuShaderModuleRelease(shader_module);
        }
    };
    error_scope.pop_and_throw_if_error();

    POSTCONDITION_MSG(shader_module != nullptr, "wgpuDeviceCreateShaderModule returned nullptr");
    m_logger->debug("Shader module created successfully");
    return ShaderModule(shader_module);
}

auto Device::create_pipeline_layout() const -> PipelineLayout {
    PRECONDITION_MSG(is_ready(), "create_pipeline_layout() called when not Ready");
    const auto& ready = std::get<ReadyState>(m_state);
    m_logger->debug("Creating empty pipeline layout");

    WGPUPipelineLayoutDescriptor layout_desc = {};
    layout_desc.bindGroupLayoutCount = 0;
    layout_desc.bindGroupLayouts = nullptr;

    ErrorScope error_scope(
        *this, {WGPUErrorFilter_Validation, WGPUErrorFilter_OutOfMemory, WGPUErrorFilter_Internal},
        m_logger->name(), "pipeline layout");
    WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(ready.device, &layout_desc);

    SCOPE_FAIL {
        m_logger->error("Failed to create pipeline layout");
        if (layout != nullptr) {
            wgpuPipelineLayoutRelease(layout);
        }
    };
    error_scope.pop_and_throw_if_error();

    POSTCONDITION_MSG(layout != nullptr, "wgpuDeviceCreatePipelineLayout returned nullptr");

    m_logger->debug("Pipeline layout created successfully");
    return PipelineLayout(layout);
}

}  // namespace pts::webgpu
