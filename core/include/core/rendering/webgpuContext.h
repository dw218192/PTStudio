#pragma once

#include <core/rendering/webgpu/asyncStateMachine.h>
#include <core/rendering/webgpu/device.h>
#include <core/rendering/webgpu/surface.h>
#include <core/rendering/webgpu/webgpu.h>
#include <core/rendering/windowing.h>

#include <memory>

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

/// State when context is initializing (device creation in flight)
struct ContextInitializingState {
    NativeViewportHandle viewport_handle{};
    Extent2D viewport_extent{};
    std::unique_ptr<pts::webgpu::Device> device;
};

/// State when context is ready and usable
struct ContextReadyState {
    pts::webgpu::Device device;
    pts::webgpu::Surface surface;

    ContextReadyState(pts::webgpu::Device d, pts::webgpu::Surface s)
        : device(std::move(d)), surface(std::move(s)) {
    }

    ContextReadyState(ContextReadyState&&) noexcept = default;
    auto operator=(ContextReadyState&&) noexcept -> ContextReadyState& = default;

    ContextReadyState(const ContextReadyState&) = delete;
    auto operator=(const ContextReadyState&) -> ContextReadyState& = delete;
};

/// State when initialization failed
struct ContextFailedState {};

/**
 * @brief WebGPU rendering context bundling device, surface, and callbacks.
 * The application owns this context and passes it to rendering backends.
 */
class WebGpuContext
    : private pts::webgpu::AsyncStateMachine<WebGpuContext, ContextInitializingState,
                                             ContextReadyState, ContextFailedState> {
    using Base = pts::webgpu::AsyncStateMachine<WebGpuContext, ContextInitializingState,
                                                ContextReadyState, ContextFailedState>;
    friend Base;

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
    // Tag type to enable make_unique with private constructor
    struct PrivateCtorTag {};

   public:
    // Constructor accessible via make_unique (use PrivateCtorTag to prevent public use)
    explicit WebGpuContext(PrivateCtorTag, std::shared_ptr<spdlog::logger> logger);

   private:
    // CRTP interface for AsyncStateMachine
    void on_tick();
    [[nodiscard]] auto is_pending() const -> bool;
    [[nodiscard]] auto wgpu_instance() const -> WGPUInstance;

    void finish_initialization();
    void set_failed();

    std::shared_ptr<spdlog::logger> m_logger;
};

}  // namespace pts::rendering
