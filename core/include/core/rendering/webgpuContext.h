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
class WebGpuContext : public pts::webgpu::AsyncStateMachine<WebGpuContext, ContextInitializingState,
                                                            ContextReadyState, ContextFailedState> {
    using Base = pts::webgpu::AsyncStateMachine<WebGpuContext, ContextInitializingState,
                                                ContextReadyState, ContextFailedState>;

   public:
    using Base::is;
    using Base::tick;

    ~WebGpuContext();

    WebGpuContext(const WebGpuContext&) = delete;
    auto operator=(const WebGpuContext&) -> WebGpuContext& = delete;

    WebGpuContext(WebGpuContext&&) noexcept;
    auto operator=(WebGpuContext&&) noexcept -> WebGpuContext&;

    /// Create a context and start async initialization. Returns Initializing state.
    /// Call tick() in a loop until is<ContextReadyState>() or is<ContextFailedState>().
    [[nodiscard]] static auto create(const IViewport& viewport,
                                     pts::LoggingManager& logging_manager)
        -> std::unique_ptr<WebGpuContext>;

    /// Access device. Only valid when is<ContextReadyState>().
    [[nodiscard]] auto device() const noexcept -> const pts::webgpu::Device&;

    /// Access surface. Only valid when is<ContextReadyState>().
    [[nodiscard]] auto surface() noexcept -> pts::webgpu::Surface&;

    /// Get surface format. Only valid when is<ContextReadyState>().
    [[nodiscard]] auto surface_format() const noexcept -> WGPUTextureFormat;

   private:
    struct PrivateCtorTag {};

   public:
    explicit WebGpuContext(PrivateCtorTag, std::shared_ptr<spdlog::logger> logger);

    // CRTP interface for AsyncStateMachine
    void on_tick();
    [[nodiscard]] auto is_pending() const -> bool;
    [[nodiscard]] auto wgpu_instance() const -> WGPUInstance;

   private:
    void finish_initialization();
    void set_failed();

    std::shared_ptr<spdlog::logger> m_logger;
};

}  // namespace pts::rendering
