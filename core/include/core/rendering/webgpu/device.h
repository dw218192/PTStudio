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

/// State of the Device during its lifecycle.
enum class DeviceState {
    Initializing,  ///< Adapter/device request in flight; usage disallowed
    Ready,         ///< Device is valid and usable
    Failed,        ///< Initialization failed; device is not usable
};

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

class Device : private AsyncStateMachine<Device, DeviceInitializingState, DeviceReadyState,
                                         DeviceFailedState> {
    using Base =
        AsyncStateMachine<Device, DeviceInitializingState, DeviceReadyState, DeviceFailedState>;
    friend Base;

   public:
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
    /// Call tick_init() until is_ready() or is_failed().
    [[nodiscard]] static auto create_async(std::shared_ptr<spdlog::logger> logger)
        -> std::unique_ptr<Device>;

    /// Process WebGPU events to advance initialization. Call until is_ready() or is_failed().
    void tick_init();

    [[nodiscard]] auto state() const noexcept -> DeviceState;
    [[nodiscard]] auto is_ready() const noexcept -> bool;
    [[nodiscard]] auto is_failed() const noexcept -> bool;
    [[nodiscard]] auto is_initializing() const noexcept -> bool;

    /// Access instance handle. Only valid when state() == Ready.
    [[nodiscard]] auto instance() const noexcept -> WGPUInstance;
    /// Access device handle. Only valid when state() == Ready.
    [[nodiscard]] auto handle() const noexcept -> WGPUDevice;
    /// Access queue handle. Only valid when state() == Ready.
    [[nodiscard]] auto queue() const noexcept -> WGPUQueue;

    [[nodiscard]] auto create_buffer(std::size_t size, WGPUBufferUsage usage) const -> Buffer;
    [[nodiscard]] auto create_shader_module_from_source(std::string_view wgsl_source) const
        -> ShaderModule;
    [[nodiscard]] auto create_pipeline_layout() const -> PipelineLayout;

   private:
    // Tag type to enable make_unique with private constructor
    struct PrivateCtorTag {};

   public:
    // Constructor accessible via make_unique (use PrivateCtorTag to prevent public use)
    explicit Device(PrivateCtorTag, std::shared_ptr<spdlog::logger> logger,
                    DeviceInitializingState init_state);

   private:
    // CRTP interface for AsyncStateMachine
    void on_tick();
    [[nodiscard]] auto is_pending() const -> bool;
    [[nodiscard]] auto wgpu_instance() const -> WGPUInstance;

    void start_adapter_request();
    void start_device_request();
    void finish_initialization();
    void set_failed();

    void release_resources();

    std::shared_ptr<spdlog::logger> m_logger;
};

}  // namespace pts::webgpu
