#include <core/rendering/webgpu/device.h>

#include <chrono>
#include <fstream>
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

Device::Device(WGPUInstance instance, WGPUDevice device, WGPUQueue queue)
    : m_instance(instance), m_device(device), m_queue(queue) {
}

Device::Device(Device&& other) noexcept
    : m_instance(other.m_instance), m_device(other.m_device), m_queue(other.m_queue) {
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

auto Device::create() -> Device {
    WGPUInstanceDescriptor instance_descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    WGPUInstance instance = wgpuCreateInstance(&instance_descriptor);
    if (instance == nullptr) {
        return {};
    }

#ifdef __EMSCRIPTEN__
    WGPUDevice device = emscripten_webgpu_get_device();
    if (device == nullptr) {
        wgpuInstanceRelease(instance);
        return {};
    }
    wgpuDeviceAddRef(device);
    WGPUQueue queue = wgpuDeviceGetQueue(device);
    wgpuQueueAddRef(queue);
    return Device(instance, device, queue);
#else
    struct AdapterRequest {
        WGPURequestAdapterStatus status = WGPURequestAdapterStatus_Error;
        WGPUAdapter adapter = nullptr;
        bool completed = false;
    };
    AdapterRequest adapter_request;
    WGPURequestAdapterOptions options = {};
    options.backendType = WGPUBackendType_Null;
    WGPURequestAdapterCallbackInfo adapter_callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    adapter_callback.mode = WGPUCallbackMode_AllowSpontaneous;
    adapter_callback.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                                   WGPUStringView, void* userdata1, void*) {
        auto* request = static_cast<AdapterRequest*>(userdata1);
        request->status = status;
        request->adapter = adapter;
        request->completed = true;
    };
    adapter_callback.userdata1 = &adapter_request;
    wgpuInstanceRequestAdapter(instance, &options, adapter_callback);
    for (int attempt = 0; attempt < 1000 && !adapter_request.completed; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (adapter_request.status != WGPURequestAdapterStatus_Success ||
        adapter_request.adapter == nullptr) {
        wgpuInstanceRelease(instance);
        return {};
    }

    struct DeviceRequest {
        WGPURequestDeviceStatus status = WGPURequestDeviceStatus_Error;
        WGPUDevice device = nullptr;
        bool completed = false;
    };
    DeviceRequest device_request;
    WGPUDeviceDescriptor device_descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    WGPURequestDeviceCallbackInfo device_callback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    device_callback.mode = WGPUCallbackMode_AllowSpontaneous;
    device_callback.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView,
                                  void* userdata1, void*) {
        auto* request = static_cast<DeviceRequest*>(userdata1);
        request->status = status;
        request->device = device;
        request->completed = true;
    };
    device_callback.userdata1 = &device_request;
    wgpuAdapterRequestDevice(adapter_request.adapter, &device_descriptor, device_callback);
    for (int attempt = 0; attempt < 1000 && !device_request.completed; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    wgpuAdapterRelease(adapter_request.adapter);
    if (device_request.status != WGPURequestDeviceStatus_Success ||
        device_request.device == nullptr) {
        wgpuInstanceRelease(instance);
        return {};
    }

    WGPUQueue queue = wgpuDeviceGetQueue(device_request.device);
    wgpuQueueAddRef(queue);
    return Device(instance, device_request.device, queue);
#endif
}

auto Device::is_valid() const noexcept -> bool {
    return m_device != nullptr;
}

auto Device::handle() const noexcept -> WGPUDevice {
    return m_device;
}

auto Device::queue() const noexcept -> WGPUQueue {
    return m_queue;
}

auto Device::create_buffer(std::size_t size, WGPUBufferUsage usage) const -> Buffer {
    if (m_device == nullptr) {
        return {};
    }
    WGPUBufferDescriptor descriptor = {};
    descriptor.size = static_cast<uint64_t>(size);
    descriptor.usage = usage;
    descriptor.mappedAtCreation = false;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(m_device, &descriptor);
    return Buffer(buffer, size);
}

auto Device::create_shader_module(std::string_view wgsl_path) const -> ShaderModule {
    if (m_device == nullptr) {
        return {};
    }
    const std::string source = load_file_contents(wgsl_path);
    if (source.empty()) {
        return {};
    }

    WGPUShaderSourceWGSL wgsl_descriptor = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl_descriptor.code = WGPUStringView{source.c_str(), source.size()};

    WGPUShaderModuleDescriptor descriptor = {};
    descriptor.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl_descriptor);
    WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(m_device, &descriptor);
    return ShaderModule(shader_module);
}

}  // namespace pts::webgpu
