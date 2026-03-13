#pragma once

#include <core/rendering/webgpu/asyncStateMachine.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>

#include <memory>
#include <string_view>
#include <utility>

namespace spdlog {
class logger;
}

namespace pts::webgpu {

/// Internal init phase within Initializing state
enum class DeviceInitPhase {
    RequestingAdapter,
    RequestingDevice,
};

/// State when device is initializing (adapter/device request in flight)
struct DeviceInitializingState {
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;

    bool adapter_request_done = false;
    bool device_request_done = false;
    WGPURequestAdapterStatus adapter_status = WGPURequestAdapterStatus_Error;
    WGPURequestDeviceStatus device_status = WGPURequestDeviceStatus_Error;

    DeviceInitPhase phase = DeviceInitPhase::RequestingAdapter;
};

/// State when device is ready and usable.
/// Move zeroes the source to prevent double-release of raw WebGPU handles.
struct DeviceReadyState {
    WGPUInstance instance = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;

    DeviceReadyState() = default;
    DeviceReadyState(WGPUInstance i, WGPUDevice d, WGPUQueue q) : instance(i), device(d), queue(q) {
    }

    DeviceReadyState(DeviceReadyState&& o) noexcept
        : instance(std::exchange(o.instance, nullptr)),
          device(std::exchange(o.device, nullptr)),
          queue(std::exchange(o.queue, nullptr)) {
    }

    auto operator=(DeviceReadyState&& o) noexcept -> DeviceReadyState& {
        instance = std::exchange(o.instance, nullptr);
        device = std::exchange(o.device, nullptr);
        queue = std::exchange(o.queue, nullptr);
        return *this;
    }

    DeviceReadyState(const DeviceReadyState&) = delete;
    auto operator=(const DeviceReadyState&) -> DeviceReadyState& = delete;
};

/// State when initialization failed
struct DeviceFailedState {};

class Device : public AsyncStateMachine<Device, DeviceInitializingState, DeviceReadyState,
                                        DeviceFailedState> {
    using Base =
        AsyncStateMachine<Device, DeviceInitializingState, DeviceReadyState, DeviceFailedState>;

   public:
    using Base::is;
    using Base::tick;
    using Base::tick_until_settled;

    /// Constructor for creating a Device with already-acquired handles.
    /// Enforces invariants: all handles must be non-null or throws std::runtime_error.
    /// Device starts in Ready state.
    explicit Device(WGPUInstance instance, WGPUDevice device, WGPUQueue queue,
                    std::shared_ptr<spdlog::logger> logger);

    Device(const Device&) = delete;
    auto operator=(const Device&) -> Device& = delete;

    Device(Device&& other) noexcept;
    auto operator=(Device&& other) noexcept -> Device&;

    ~Device();

    /// Blocking factory: creates device with error callbacks registered.
    /// Internally uses create_async() + tick_until_settled(). Throws on failure.
    [[nodiscard]] static auto create(std::shared_ptr<spdlog::logger> logger) -> Device;

    /// Async factory: starts device creation and returns Initializing device.
    /// Call tick() until is<DeviceReadyState>() or is<DeviceFailedState>().
    [[nodiscard]] static auto create_async(std::shared_ptr<spdlog::logger> logger)
        -> std::unique_ptr<Device>;

    /// Access instance handle. Only valid when is<DeviceReadyState>().
    [[nodiscard]] auto instance() const noexcept -> WGPUInstance;
    /// Access device handle. Only valid when is<DeviceReadyState>().
    [[nodiscard]] auto handle() const noexcept -> WGPUDevice;
    /// Access queue handle. Only valid when is<DeviceReadyState>().
    [[nodiscard]] auto queue() const noexcept -> WGPUQueue;

    [[nodiscard]] auto create_buffer(std::size_t size, WGPUBufferUsage usage) const -> Buffer;
    [[nodiscard]] auto create_shader_module_from_source(std::string_view wgsl_source) const
        -> ShaderModule;
    [[nodiscard]] auto create_pipeline_layout() const -> PipelineLayout;

   private:
    struct PrivateCtorTag {};

   public:
    explicit Device(PrivateCtorTag, std::shared_ptr<spdlog::logger> logger,
                    DeviceInitializingState init_state);

    // CRTP interface for AsyncStateMachine
    void on_tick();
    [[nodiscard]] auto is_pending() const -> bool;
    [[nodiscard]] auto wgpu_instance() const -> WGPUInstance;

   private:
    void start_adapter_request();
    void start_device_request();
    void finish_initialization();
    void set_failed();

    void release_resources();

    std::shared_ptr<spdlog::logger> m_logger;
};

}  // namespace pts::webgpu
