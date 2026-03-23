#pragma once

#include <core/rendering/webgpu/webgpu.h>

namespace pts::webgpu {
class Device;
}

namespace pts::rendering {

/// GPU textures holding the LTC (Linearly Transformed Cosines) lookup tables.
/// Two 64x64 textures: M^(-1) matrix parameters (RGBA16Float) and
/// Fresnel-weighted amplitude (RG16Float), plus a bilinear-clamp sampler.
class LtcTextures {
   public:
    void init(const webgpu::Device& device);

    WGPUTextureView mat_view() const noexcept {
        return m_mat_view;
    }
    WGPUTextureView amp_view() const noexcept {
        return m_amp_view;
    }
    WGPUSampler sampler() const noexcept {
        return m_sampler;
    }

    ~LtcTextures();
    LtcTextures() = default;
    LtcTextures(const LtcTextures&) = delete;
    LtcTextures& operator=(const LtcTextures&) = delete;
    LtcTextures(LtcTextures&&) noexcept;
    LtcTextures& operator=(LtcTextures&&) noexcept;

   private:
    void release();

    WGPUTexture m_mat_tex = nullptr;
    WGPUTextureView m_mat_view = nullptr;
    WGPUTexture m_amp_tex = nullptr;
    WGPUTextureView m_amp_view = nullptr;
    WGPUSampler m_sampler = nullptr;
};

}  // namespace pts::rendering
