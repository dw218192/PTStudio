#pragma once

#include <core/rendering/webgpu/device.h>
#include <core/rendering/webgpu/surface.h>
#include <core/rendering/webgpu/webgpu.h>
#include <core/rendering/windowing.h>

#include <functional>
#include <memory>

namespace pts {
class LoggingManager;
}

namespace pts::rendering {

/**
 * @brief WebGPU rendering context bundling device, surface, and callbacks.
 * The application owns this context and passes it to rendering backends.
 */
class WebGpuContext {
   public:
    // Constructor - prefer using create() factory method
    explicit WebGpuContext(pts::webgpu::Device device, pts::webgpu::Surface surface);
    ~WebGpuContext();

    WebGpuContext(const WebGpuContext&) = delete;
    auto operator=(const WebGpuContext&) -> WebGpuContext& = delete;

    WebGpuContext(WebGpuContext&&) noexcept;
    auto operator=(WebGpuContext&&) noexcept -> WebGpuContext&;

    [[nodiscard]] static auto create(const IViewport& viewport,
                                     pts::LoggingManager& logging_manager)
        -> std::unique_ptr<WebGpuContext>;

    [[nodiscard]] auto is_valid() const noexcept -> bool;
    [[nodiscard]] auto device() const noexcept -> const pts::webgpu::Device&;
    [[nodiscard]] auto surface() noexcept -> pts::webgpu::Surface&;
    [[nodiscard]] auto surface_format() const noexcept -> WGPUTextureFormat;

   private:
    // invariants:
    // - m_device is always valid if the class is constructed successfully
    // - m_surface is always valid if the class is constructed successfully

    pts::webgpu::Device m_device;
    pts::webgpu::Surface m_surface;
};

}  // namespace pts::rendering
