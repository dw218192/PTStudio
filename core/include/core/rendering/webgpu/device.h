#pragma once

#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/texture.h>

#include <memory>
#include <string>
#include <string_view>
#include <variant>

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

class Device {
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
    /// Internally uses create_async() + tick_init() loop. Throws on failure.
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
    /// Internal init phase within Initializing state
    enum class InitPhase {
        RequestingAdapter,
        RequestingDevice,
    };

    /// State when device is initializing (adapter/device request in flight)
    struct InitializingState {
        WGPUInstance instance = nullptr;
        WGPUAdapter adapter = nullptr;
        WGPUDevice device = nullptr;  // Set by callback before transitioning to Ready

        bool adapter_request_done = false;
        bool device_request_done = false;
        WGPURequestAdapterStatus adapter_status = WGPURequestAdapterStatus_Error;
        WGPURequestDeviceStatus device_status = WGPURequestDeviceStatus_Error;

        InitPhase phase = InitPhase::RequestingAdapter;
    };

    /// State when device is ready and usable
    struct ReadyState {
        WGPUInstance instance = nullptr;
        WGPUDevice device = nullptr;
        WGPUQueue queue = nullptr;
    };

    /// State when initialization failed
    struct FailedState {};

    using State = std::variant<InitializingState, ReadyState, FailedState>;

    // Tag type to enable make_unique with private constructor
    struct PrivateCtorTag {};

   public:
    // Constructor accessible via make_unique (use PrivateCtorTag to prevent public use)
    explicit Device(PrivateCtorTag, std::shared_ptr<spdlog::logger> logger,
                    InitializingState init_state);

   private:
    void start_adapter_request();
    void start_device_request();
    void finish_initialization();
    void set_failed();

    // Helper to release resources in current state
    void release_resources();

    // Helper to get state enum from variant
    [[nodiscard]] auto get_state_enum() const noexcept -> DeviceState;

    State m_state = FailedState{};
    std::shared_ptr<spdlog::logger> m_logger;
};

}  // namespace pts::webgpu
