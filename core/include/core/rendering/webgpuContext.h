#pragma once

#include <core/rendering/webgpu/device.h>
#include <core/rendering/webgpu/surface.h>
#include <core/rendering/webgpu/webgpu.h>
#include <core/rendering/windowing.h>

#include <functional>
#include <memory>
#include <variant>

namespace spdlog {
class logger;
}

namespace pts {
class LoggingManager;
}

namespace pts::rendering {

/// State of the WebGpuContext during its lifecycle.
enum class WebGpuContextState {
    Initializing,  ///< Adapter/device request in flight; rendering disallowed
    Ready,         ///< Device and surface are valid and usable
    Failed,        ///< Initialization failed; context is not usable
};

/**
 * @brief WebGPU rendering context bundling device, surface, and callbacks.
 * The application owns this context and passes it to rendering backends.
 *
 * State is modeled using std::variant to enforce invariants at the type level:
 * - InitializingState: async request in flight; device/surface not yet available.
 * - ReadyState: device and surface are valid and usable.
 * - FailedState: context is not usable.
 */
class WebGpuContext {
   public:
    ~WebGpuContext();

    WebGpuContext(const WebGpuContext&) = delete;
    auto operator=(const WebGpuContext&) -> WebGpuContext& = delete;

    WebGpuContext(WebGpuContext&&) noexcept;
    auto operator=(WebGpuContext&&) noexcept -> WebGpuContext&;

    /// Create a context and start async initialization. Returns Initializing state.
    /// Call tick_init() in a loop until is_ready() or is_failed().
    [[nodiscard]] static auto create(const IViewport& viewport,
                                     pts::LoggingManager& logging_manager)
        -> std::unique_ptr<WebGpuContext>;

    /// Process WebGPU events to advance initialization. Call until is_ready() or is_failed().
    void tick_init();

    [[nodiscard]] auto state() const noexcept -> WebGpuContextState;
    [[nodiscard]] auto is_ready() const noexcept -> bool;
    [[nodiscard]] auto is_failed() const noexcept -> bool;
    [[nodiscard]] auto is_initializing() const noexcept -> bool;

    /// Access device. Only valid when state() == Ready.
    [[nodiscard]] auto device() const noexcept -> const pts::webgpu::Device&;

    /// Access surface. Only valid when state() == Ready.
    [[nodiscard]] auto surface() noexcept -> pts::webgpu::Surface&;

    /// Get surface format. Only valid when state() == Ready.
    [[nodiscard]] auto surface_format() const noexcept -> WGPUTextureFormat;

   private:
    /// Internal init phase within Initializing state
    enum class InitPhase {
        RequestingAdapter,
        RequestingDevice,
        CreatingSurface,
    };

    /// State when context is initializing (adapter/device request in flight)
    struct InitializingState {
        // Viewport data for surface creation
        NativeViewportHandle viewport_handle{};
        Extent2D viewport_extent{};

        // WebGPU handles during init
        WGPUInstance instance = nullptr;
        WGPUAdapter adapter = nullptr;
        WGPUDevice device = nullptr;
        WGPUQueue queue = nullptr;

        // Request completion flags (set synchronously by callbacks during ProcessEvents)
        bool adapter_request_done = false;
        bool device_request_done = false;
        WGPURequestAdapterStatus adapter_status = WGPURequestAdapterStatus_Error;
        WGPURequestDeviceStatus device_status = WGPURequestDeviceStatus_Error;

        InitPhase phase = InitPhase::RequestingAdapter;
    };

    /// State when context is ready and usable
    struct ReadyState {
        pts::webgpu::Device device;
        pts::webgpu::Surface surface;

        ReadyState(pts::webgpu::Device d, pts::webgpu::Surface s)
            : device(std::move(d)), surface(std::move(s)) {
        }
    };

    /// State when initialization failed
    struct FailedState {};

    using State = std::variant<InitializingState, ReadyState, FailedState>;

    // Tag type to enable make_unique with private constructor
    struct PrivateCtorTag {};

   public:
    // Constructor accessible via make_unique (use PrivateCtorTag to prevent public use)
    explicit WebGpuContext(PrivateCtorTag, std::shared_ptr<spdlog::logger> logger);

   private:
    // Transition helpers
    void start_adapter_request();
    void start_device_request();
    void finish_initialization();
    void set_failed();

    // Helper to release resources in current state
    void release_resources();

    // Helper to get state enum from variant
    [[nodiscard]] auto get_state_enum() const noexcept -> WebGpuContextState;

    State m_state = FailedState{};
    std::shared_ptr<spdlog::logger> m_logger;
};

}  // namespace pts::rendering
