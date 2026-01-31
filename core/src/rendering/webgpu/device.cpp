#include <core/diagnostics.h>
#include <core/rendering/webgpu/device.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef __EMSCRIPTEN__
#include <emscripten/html5_webgpu.h>
#endif

namespace pts::webgpu {
namespace {
auto load_file_contents(std::string_view path) -> std::string {
    std::ifstream file(std::string(path), std::ios::binary);
    if (!file) {
        return {};
    }
    file.seekg(0, std::ios::end);
    std::string contents;
    contents.resize(static_cast<size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    file.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    return contents;
}
}  // namespace

Device::Device(WGPUInstance instance, WGPUDevice device, WGPUQueue queue,
               std::shared_ptr<spdlog::logger> logger)
    : m_instance(instance), m_device(device), m_queue(queue), m_logger(std::move(logger)) {
    INVARIANT_MSG(m_instance != nullptr, "instance handle is null");
    INVARIANT_MSG(m_device != nullptr, "device handle is null");
    INVARIANT_MSG(m_queue != nullptr, "queue handle is null");
    INVARIANT_MSG(m_logger != nullptr, "logger is null");

    m_logger->debug("Device constructed successfully (device={}, queue={})",
                    static_cast<void*>(m_device), static_cast<void*>(m_queue));
}

Device::Device(Device&& other) noexcept
    : m_instance(other.m_instance),
      m_device(other.m_device),
      m_queue(other.m_queue),
      m_logger(std::move(other.m_logger)) {
    other.m_instance = nullptr;
    other.m_device = nullptr;
    other.m_queue = nullptr;
}

auto Device::operator=(Device&& other) noexcept -> Device& {
    if (this != &other) {
        if (m_queue != nullptr) {
            wgpuQueueRelease(m_queue);
        }
        if (m_device != nullptr) {
            wgpuDeviceRelease(m_device);
        }
        if (m_instance != nullptr) {
            wgpuInstanceRelease(m_instance);
        }
        m_instance = other.m_instance;
        m_device = other.m_device;
        m_queue = other.m_queue;
        m_logger = std::move(other.m_logger);
        other.m_instance = nullptr;
        other.m_device = nullptr;
        other.m_queue = nullptr;
    }
    return *this;
}

Device::~Device() {
    if (m_queue != nullptr) {
        wgpuQueueRelease(m_queue);
    }
    if (m_device != nullptr) {
        wgpuDeviceRelease(m_device);
    }
    if (m_instance != nullptr) {
        wgpuInstanceRelease(m_instance);
    }
}

auto Device::create(std::shared_ptr<spdlog::logger> logger) -> Device {
    PRECONDITION_MSG(logger != nullptr, "logger is null");

    logger->info("Creating WebGPU device...");

    WGPUInstanceDescriptor instance_descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    WGPUInstance instance = wgpuCreateInstance(&instance_descriptor);
    if (instance == nullptr) {
        logger->error("Failed to create WebGPU instance");
        throw std::runtime_error("Failed to create WebGPU instance");
    }

#ifdef __EMSCRIPTEN__
    logger->debug("Using Emscripten/Browser WebGPU backend");
#else
    logger->debug("Using Dawn/Native WebGPU backend");
#endif

    struct AdapterRequest {
        WGPURequestAdapterStatus status = WGPURequestAdapterStatus_Error;
        WGPUAdapter adapter = nullptr;
        bool completed = false;
    };
    AdapterRequest adapter_request;
    WGPURequestAdapterOptions options = {};
    options.backendType = WGPUBackendType_Undefined;
    WGPURequestAdapterCallbackInfo adapter_callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    adapter_callback.mode = WGPUCallbackMode_AllowSpontaneous;
    adapter_callback.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                                   WGPUStringView, void* userdata1, void*) {
        PRECONDITION(userdata1 != nullptr);
        auto* request = static_cast<AdapterRequest*>(userdata1);
        request->status = status;
        request->adapter = adapter;
        request->completed = true;
    };
    adapter_callback.userdata1 = &adapter_request;

    logger->debug("Requesting WebGPU adapter...");
    wgpuInstanceRequestAdapter(instance, &options, adapter_callback);
    for (int attempt = 0; attempt < 1000 && !adapter_request.completed; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (!adapter_request.completed) {
        logger->error("WebGPU adapter request timed out");
        wgpuInstanceRelease(instance);
        throw std::runtime_error("WebGPU adapter request timed out");
    }

    if (adapter_request.status != WGPURequestAdapterStatus_Success) {
        logger->error("Failed to request WebGPU adapter (status: {})",
                      static_cast<int>(adapter_request.status));
        wgpuInstanceRelease(instance);
        throw std::runtime_error("Failed to request WebGPU adapter");
    }

    if (adapter_request.adapter == nullptr) {
        logger->error("WebGPU adapter is null");
        wgpuInstanceRelease(instance);
        throw std::runtime_error("WebGPU adapter is null");
    }

    logger->debug("WebGPU adapter acquired successfully");

    struct DeviceRequest {
        WGPURequestDeviceStatus status = WGPURequestDeviceStatus_Error;
        WGPUDevice device = nullptr;
        bool completed = false;
    };
    DeviceRequest device_request;

    // Setup device descriptor with error callbacks
    WGPUDeviceDescriptor device_descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;

    // Configure device lost callback
    WGPUDeviceLostCallbackInfo device_lost_callback = WGPU_DEVICE_LOST_CALLBACK_INFO_INIT;
    device_lost_callback.mode = WGPUCallbackMode_AllowSpontaneous;
    device_lost_callback.callback = [](WGPUDevice const*, WGPUDeviceLostReason reason,
                                       WGPUStringView message, void* userdata1, void*) {
        PRECONDITION(userdata1 != nullptr);
        auto* log = static_cast<spdlog::logger*>(userdata1);
        const char* reason_str = "Unknown";
        spdlog::level::level_enum level = spdlog::level::err;
        switch (reason) {
            case WGPUDeviceLostReason_Unknown:
                reason_str = "Unknown";
                break;
            case WGPUDeviceLostReason_Destroyed:
                reason_str = "Destroyed";
                level = spdlog::level::info;
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
        log->log(level, "[WebGPU Device Lost] Reason: {}, Message: {}", reason_str,
                 message.data ? std::string_view(message.data, message.length)
                              : std::string_view("(no message)"));
    };
    device_lost_callback.userdata1 = logger.get();
    device_descriptor.deviceLostCallbackInfo = device_lost_callback;

    // Configure uncaptured error callback
    WGPUUncapturedErrorCallbackInfo uncaptured_error_callback =
        WGPU_UNCAPTURED_ERROR_CALLBACK_INFO_INIT;
    uncaptured_error_callback.callback = [](WGPUDevice const*, WGPUErrorType type,
                                            WGPUStringView message, void* userdata1, void*) {
        if (type == WGPUErrorType_NoError) {
            return;  // Don't log "no error"
        }
        PRECONDITION(userdata1 != nullptr);
        auto* log = static_cast<spdlog::logger*>(userdata1);
        const char* error_type_name = "Unknown";
        switch (type) {
            case WGPUErrorType_Validation:
                error_type_name = "Validation";
                break;
            case WGPUErrorType_OutOfMemory:
                error_type_name = "OutOfMemory";
                break;
            case WGPUErrorType_Internal:
                error_type_name = "Internal";
                break;
            case WGPUErrorType_Unknown:
            default:
                error_type_name = "Unknown";
                break;
            case WGPUErrorType_NoError:
                return;
        }
        log->error("[WebGPU Uncaptured Error] Type: {}, Message: {}", error_type_name,
                   message.data ? std::string_view(message.data, message.length)
                                : std::string_view("(no message)"));
    };
    uncaptured_error_callback.userdata1 = logger.get();
    device_descriptor.uncapturedErrorCallbackInfo = uncaptured_error_callback;

    WGPURequestDeviceCallbackInfo device_callback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    device_callback.mode = WGPUCallbackMode_AllowSpontaneous;
    device_callback.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView,
                                  void* userdata1, void*) {
        PRECONDITION(userdata1 != nullptr);
        auto* request = static_cast<DeviceRequest*>(userdata1);
        request->status = status;
        request->device = device;
        request->completed = true;
    };
    device_callback.userdata1 = &device_request;

    logger->debug("Requesting WebGPU device...");
    wgpuAdapterRequestDevice(adapter_request.adapter, &device_descriptor, device_callback);
    for (int attempt = 0; attempt < 1000 && !device_request.completed; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    wgpuAdapterRelease(adapter_request.adapter);

    if (!device_request.completed) {
        logger->error("WebGPU device request timed out");
        wgpuInstanceRelease(instance);
        throw std::runtime_error("WebGPU device request timed out");
    }

    if (device_request.status != WGPURequestDeviceStatus_Success) {
        logger->error("Failed to request WebGPU device (status: {})",
                      static_cast<int>(device_request.status));
        wgpuInstanceRelease(instance);
        throw std::runtime_error("Failed to request WebGPU device");
    }

    if (device_request.device == nullptr) {
        logger->error("WebGPU device is null");
        wgpuInstanceRelease(instance);
        throw std::runtime_error("WebGPU device is null");
    }

    logger->debug("WebGPU device acquired successfully");

    WGPUQueue queue = wgpuDeviceGetQueue(device_request.device);
    if (queue == nullptr) {
        logger->error("Failed to get WebGPU queue");
        wgpuDeviceRelease(device_request.device);
        wgpuInstanceRelease(instance);
        throw std::runtime_error("Failed to get WebGPU queue");
    }
    wgpuQueueAddRef(queue);
    logger->debug("WebGPU queue acquired successfully");

#ifdef __EMSCRIPTEN__
    logger->info("WebGPU device created successfully (Emscripten/Browser backend)");
#else
    logger->info("WebGPU device created successfully (Dawn/Native backend)");
#endif
    logger->debug("Device lost and uncaptured error callbacks are registered");
    return Device(instance, device_request.device, queue, logger);
}

auto Device::instance() const noexcept -> WGPUInstance {
    return m_instance;
}

auto Device::handle() const noexcept -> WGPUDevice {
    return m_device;
}

auto Device::queue() const noexcept -> WGPUQueue {
    return m_queue;
}

auto Device::create_buffer(std::size_t size, WGPUBufferUsage usage) const -> Buffer {
    // Device is always valid due to enforced invariants
    m_logger->debug("Creating buffer (size={}, usage={})", size, usage);

    WGPUBufferDescriptor descriptor = {};
    descriptor.size = static_cast<uint64_t>(size);
    descriptor.usage = usage;
    descriptor.mappedAtCreation = false;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(m_device, &descriptor);

    if (buffer == nullptr) {
        m_logger->error("Failed to create WebGPU buffer (size={}, usage={})", size, usage);
        throw std::runtime_error("Failed to create WebGPU buffer");
    }

    m_logger->debug("Buffer created successfully (handle={})", static_cast<void*>(buffer));
    return Buffer(buffer, size);
}

auto Device::create_shader_module(std::string_view wgsl_path) const -> ShaderModule {
    // Device is always valid due to enforced invariants
    m_logger->debug("Loading shader module from: {}", wgsl_path);

    const std::string source = load_file_contents(wgsl_path);
    if (source.empty()) {
        m_logger->error("Failed to load shader source from: {}", wgsl_path);
        throw std::runtime_error("Failed to load shader source file");
    }

    m_logger->debug("Shader source loaded ({} bytes)", source.size());

    WGPUShaderSourceWGSL wgsl_descriptor = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl_descriptor.code = WGPUStringView{source.c_str(), source.size()};

    WGPUShaderModuleDescriptor descriptor = {};
    descriptor.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl_descriptor);
    WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(m_device, &descriptor);

    if (shader_module == nullptr) {
        m_logger->error("Failed to create shader module from: {}", wgsl_path);
        throw std::runtime_error("Failed to create WebGPU shader module");
    }

    m_logger->debug("Shader module created successfully");
    return ShaderModule(shader_module);
}

auto Device::create_shader_module_from_source(std::string_view wgsl_source) const -> ShaderModule {
    // Device is always valid due to enforced invariants
    m_logger->debug("Creating shader module from source ({} bytes)", wgsl_source.size());

    if (wgsl_source.empty()) {
        m_logger->error("Shader source is empty");
        throw std::runtime_error("Shader source is empty");
    }

    WGPUShaderSourceWGSL wgsl_descriptor = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl_descriptor.code = WGPUStringView{wgsl_source.data(), wgsl_source.size()};

    WGPUShaderModuleDescriptor descriptor = {};
    descriptor.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl_descriptor);
    WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(m_device, &descriptor);

    if (shader_module == nullptr) {
        m_logger->error("Failed to create shader module from source");
        throw std::runtime_error("Failed to create WebGPU shader module");
    }

    m_logger->debug("Shader module created successfully");
    return ShaderModule(shader_module);
}

auto Device::create_pipeline_layout() const -> PipelineLayout {
    // Device is always valid due to enforced invariants
    m_logger->debug("Creating empty pipeline layout");

    WGPUPipelineLayoutDescriptor layout_desc = {};
    layout_desc.bindGroupLayoutCount = 0;
    layout_desc.bindGroupLayouts = nullptr;

    WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(m_device, &layout_desc);

    if (layout == nullptr) {
        m_logger->error("Failed to create pipeline layout");
        throw std::runtime_error("Failed to create WebGPU pipeline layout");
    }

    m_logger->debug("Pipeline layout created successfully");
    return PipelineLayout(layout);
}

}  // namespace pts::webgpu
