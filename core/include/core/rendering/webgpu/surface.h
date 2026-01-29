#pragma once

#include <core/rendering/webgpu/webgpu.h>
#include <core/rendering/windowing.h>

namespace pts::webgpu {
class Device;

class Surface {
   public:
    Surface() = default;
    explicit Surface(WGPUSurface surface, WGPUDevice device, WGPUTextureFormat format,
                     WGPUTextureUsage usage, WGPUPresentMode present_mode,
                     WGPUCompositeAlphaMode alpha_mode, rendering::Extent2D extent);

    Surface(const Surface&) = delete;
    auto operator=(const Surface&) -> Surface& = delete;

    Surface(Surface&& other) noexcept;
    auto operator=(Surface&& other) noexcept -> Surface&;

    ~Surface();

    [[nodiscard]] static auto create(const Device& device,
                                     const rendering::NativeViewportHandle& handle,
                                     rendering::Extent2D extent) -> Surface;
    [[nodiscard]] auto is_valid() const noexcept -> bool;
    [[nodiscard]] auto format() const noexcept -> WGPUTextureFormat;
    [[nodiscard]] auto extent() const noexcept -> rendering::Extent2D;

    void resize(rendering::Extent2D extent);
    [[nodiscard]] auto acquire_texture_view() -> WGPUTextureView;
    void present();

   private:
    void configure(uint32_t width, uint32_t height);

    WGPUSurface m_surface = nullptr;
    WGPUDevice m_device = nullptr;
    WGPUTextureFormat m_format = WGPUTextureFormat_Undefined;
    WGPUTextureUsage m_usage = WGPUTextureUsage_RenderAttachment;
    WGPUPresentMode m_present_mode = WGPUPresentMode_Fifo;
    WGPUCompositeAlphaMode m_alpha_mode = WGPUCompositeAlphaMode_Auto;
    WGPUTexture m_current_texture = nullptr;
    WGPUTextureView m_current_view = nullptr;
    bool m_present_pending = false;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
};

}  // namespace pts::webgpu
