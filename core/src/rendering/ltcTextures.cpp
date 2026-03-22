#include <core/diagnostics.h>
#include <core/rendering/ltcData.h>
#include <core/rendering/ltcTextures.h>
#include <core/rendering/webgpu/device.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace pts::rendering {

namespace {

uint16_t float_to_half(float f) {
    // IEEE 754 float32 → float16 conversion
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }
    if (exponent == 0xFF - 127 + 15) {
        if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7C00);  // inf
        return static_cast<uint16_t>(sign | 0x7C00 | (mantissa >> 13));  // nan
    }
    if (exponent > 30) return static_cast<uint16_t>(sign | 0x7C00);  // overflow → inf

    return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

}  // namespace

void LtcTextures::init(const webgpu::Device& device) {
    release();

    constexpr uint32_t n = static_cast<uint32_t>(k_ltc_size);
    auto dev = device.handle();
    auto queue = device.queue();

    // --- M^(-1) matrix texture: RGBA16Float ---
    {
        WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        desc.size = {n, n, 1};
        desc.format = WGPUTextureFormat_RGBA16Float;
        desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding |
                                                   WGPUTextureUsage_CopyDst);
        desc.mipLevelCount = 1;
        desc.sampleCount = 1;
        desc.dimension = WGPUTextureDimension_2D;
        m_mat_tex = wgpuDeviceCreateTexture(dev, &desc);
        ASSERT_MSG(m_mat_tex, "Failed to create LTC matrix texture");

        // Convert float32 → float16
        std::vector<uint16_t> half_data(n * n * 4);
        for (size_t i = 0; i < n * n * 4; ++i) {
            half_data[i] = float_to_half(k_ltc_mat[i]);
        }

        WGPUTexelCopyBufferLayout layout = {};
        layout.offset = 0;
        layout.bytesPerRow = n * 4 * sizeof(uint16_t);  // 4 channels × 2 bytes
        layout.rowsPerImage = n;

        WGPUTexelCopyTextureInfo dest = {};
        dest.texture = m_mat_tex;
        dest.mipLevel = 0;
        dest.origin = {0, 0, 0};
        dest.aspect = WGPUTextureAspect_All;

        WGPUExtent3D extent = {n, n, 1};
        wgpuQueueWriteTexture(queue, &dest, half_data.data(), half_data.size() * sizeof(uint16_t),
                              &layout, &extent);

        WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
        view_desc.format = WGPUTextureFormat_RGBA16Float;
        view_desc.dimension = WGPUTextureViewDimension_2D;
        view_desc.mipLevelCount = 1;
        view_desc.arrayLayerCount = 1;
        m_mat_view = wgpuTextureCreateView(m_mat_tex, &view_desc);
        ASSERT_MSG(m_mat_view, "Failed to create LTC matrix texture view");
    }

    // --- Amplitude texture: RG16Float ---
    {
        WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        desc.size = {n, n, 1};
        desc.format = WGPUTextureFormat_RG16Float;
        desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding |
                                                   WGPUTextureUsage_CopyDst);
        desc.mipLevelCount = 1;
        desc.sampleCount = 1;
        desc.dimension = WGPUTextureDimension_2D;
        m_amp_tex = wgpuDeviceCreateTexture(dev, &desc);
        ASSERT_MSG(m_amp_tex, "Failed to create LTC amplitude texture");

        std::vector<uint16_t> half_data(n * n * 2);
        for (size_t i = 0; i < n * n * 2; ++i) {
            half_data[i] = float_to_half(k_ltc_amp[i]);
        }

        WGPUTexelCopyBufferLayout layout = {};
        layout.offset = 0;
        layout.bytesPerRow = n * 2 * sizeof(uint16_t);  // 2 channels × 2 bytes
        layout.rowsPerImage = n;

        WGPUTexelCopyTextureInfo dest = {};
        dest.texture = m_amp_tex;
        dest.mipLevel = 0;
        dest.origin = {0, 0, 0};
        dest.aspect = WGPUTextureAspect_All;

        WGPUExtent3D extent = {n, n, 1};
        wgpuQueueWriteTexture(queue, &dest, half_data.data(), half_data.size() * sizeof(uint16_t),
                              &layout, &extent);

        WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
        view_desc.format = WGPUTextureFormat_RG16Float;
        view_desc.dimension = WGPUTextureViewDimension_2D;
        view_desc.mipLevelCount = 1;
        view_desc.arrayLayerCount = 1;
        m_amp_view = wgpuTextureCreateView(m_amp_tex, &view_desc);
        ASSERT_MSG(m_amp_view, "Failed to create LTC amplitude texture view");
    }

    // --- Bilinear-clamp sampler ---
    {
        WGPUSamplerDescriptor desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
        desc.magFilter = WGPUFilterMode_Linear;
        desc.minFilter = WGPUFilterMode_Linear;
        desc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
        desc.addressModeU = WGPUAddressMode_ClampToEdge;
        desc.addressModeV = WGPUAddressMode_ClampToEdge;
        desc.addressModeW = WGPUAddressMode_ClampToEdge;
        m_sampler = wgpuDeviceCreateSampler(dev, &desc);
        ASSERT_MSG(m_sampler, "Failed to create LTC sampler");
    }
}

void LtcTextures::release() {
    if (m_sampler) wgpuSamplerRelease(m_sampler);
    if (m_amp_view) wgpuTextureViewRelease(m_amp_view);
    if (m_amp_tex) wgpuTextureRelease(m_amp_tex);
    if (m_mat_view) wgpuTextureViewRelease(m_mat_view);
    if (m_mat_tex) wgpuTextureRelease(m_mat_tex);
    m_sampler = nullptr;
    m_amp_view = nullptr;
    m_amp_tex = nullptr;
    m_mat_view = nullptr;
    m_mat_tex = nullptr;
}

LtcTextures::~LtcTextures() {
    release();
}

LtcTextures::LtcTextures(LtcTextures&& o) noexcept
    : m_mat_tex(std::exchange(o.m_mat_tex, nullptr)),
      m_mat_view(std::exchange(o.m_mat_view, nullptr)),
      m_amp_tex(std::exchange(o.m_amp_tex, nullptr)),
      m_amp_view(std::exchange(o.m_amp_view, nullptr)),
      m_sampler(std::exchange(o.m_sampler, nullptr)) {
}

LtcTextures& LtcTextures::operator=(LtcTextures&& o) noexcept {
    if (this != &o) {
        release();
        m_mat_tex = std::exchange(o.m_mat_tex, nullptr);
        m_mat_view = std::exchange(o.m_mat_view, nullptr);
        m_amp_tex = std::exchange(o.m_amp_tex, nullptr);
        m_amp_view = std::exchange(o.m_amp_view, nullptr);
        m_sampler = std::exchange(o.m_sampler, nullptr);
    }
    return *this;
}

}  // namespace pts::rendering
