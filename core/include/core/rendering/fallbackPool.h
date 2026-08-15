#pragma once

#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace pts::webgpu {
class Device;
}

namespace pts::rendering {

/// Lazily creates shared 1x1 fallback textures and zero-filled buffers.
/// Depth formats get value 1.0, color formats get white (1,1,1,1).
class FallbackPool {
   public:
    explicit FallbackPool(const webgpu::Device& device);
    ~FallbackPool();

    FallbackPool(const FallbackPool&) = delete;
    FallbackPool& operator=(const FallbackPool&) = delete;

    /// Get or create a 1x1 fallback texture view for the given format/dimension.
    WGPUTextureView view(WGPUTextureFormat format, WGPUTextureViewDimension dim);

    /// Get or create a zero-filled fallback buffer of at least min_size bytes.
    WGPUBuffer buffer(uint64_t min_size);

   private:
    const webgpu::Device* m_device;

    struct TextureEntry {
        WGPUTexture texture = nullptr;
        WGPUTextureView view = nullptr;
    };

    // Key: (format << 8) | dimension
    std::unordered_map<uint32_t, TextureEntry> m_textures;

    struct BufferEntry {
        WGPUBuffer buffer = nullptr;
        uint64_t size = 0;
    };
    std::vector<BufferEntry> m_buffers;

    static uint32_t make_key(WGPUTextureFormat format, WGPUTextureViewDimension dim);
};

}  // namespace pts::rendering
