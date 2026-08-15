#include <core/diagnostics.h>
#include <core/rendering/fallbackPool.h>
#include <core/rendering/webgpu/device.h>

namespace pts::rendering {

static bool is_depth_format(WGPUTextureFormat fmt) {
    switch (fmt) {
        case WGPUTextureFormat_Depth16Unorm:
        case WGPUTextureFormat_Depth24Plus:
        case WGPUTextureFormat_Depth24PlusStencil8:
        case WGPUTextureFormat_Depth32Float:
        case WGPUTextureFormat_Depth32FloatStencil8:
            return true;
        default:
            return false;
    }
}

uint32_t FallbackPool::make_key(WGPUTextureFormat format, WGPUTextureViewDimension dim) {
    return (static_cast<uint32_t>(format) << 8) | static_cast<uint32_t>(dim);
}

FallbackPool::FallbackPool(const webgpu::Device& device) : m_device(&device) {
}

FallbackPool::~FallbackPool() {
    for (auto& [key, entry] : m_textures) {
        if (entry.view) wgpuTextureViewRelease(entry.view);
        if (entry.texture) wgpuTextureRelease(entry.texture);
    }
    for (auto& entry : m_buffers) {
        if (entry.buffer) {
            wgpuBufferDestroy(entry.buffer);
            wgpuBufferRelease(entry.buffer);
        }
    }
}

WGPUTextureView FallbackPool::view(WGPUTextureFormat format, WGPUTextureViewDimension dim) {
    auto key = make_key(format, dim);
    auto it = m_textures.find(key);
    if (it != m_textures.end()) {
        return it->second.view;
    }

    // Determine array layer count based on dimension
    uint32_t layers = 1;
    auto tex_view_dim = dim;
    if (dim == WGPUTextureViewDimension_Cube) {
        layers = 6;
    } else if (dim == WGPUTextureViewDimension_2DArray) {
        layers = 1;  // minimum for array view
    }

    bool depth = is_depth_format(format);

    WGPUTextureDescriptor tex_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    tex_desc.size = {1, 1, layers};
    tex_desc.format = format;
    tex_desc.mipLevelCount = 1;
    tex_desc.sampleCount = 1;
    tex_desc.dimension = WGPUTextureDimension_2D;

    if (depth) {
        // Depth textures cannot be CopyDst -- create render-attachment-only
        tex_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding |
                                                       WGPUTextureUsage_RenderAttachment);
    } else {
        tex_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding |
                                                       WGPUTextureUsage_CopyDst);
    }

    auto texture = wgpuDeviceCreateTexture(m_device->handle(), &tex_desc);
    INVARIANT_MSG(texture, "FallbackPool: failed to create fallback texture");

    if (!depth) {
        // Fill color textures with white
        uint8_t white[4] = {255, 255, 255, 255};
        WGPUTexelCopyBufferLayout layout = {};
        layout.bytesPerRow = 256;  // WebGPU minimum
        layout.rowsPerImage = 1;
        WGPUTexelCopyTextureInfo dest = {};
        dest.texture = texture;
        dest.aspect = WGPUTextureAspect_All;
        WGPUExtent3D extent = {1, 1, 1};
        // Fill each layer
        for (uint32_t i = 0; i < layers; ++i) {
            dest.origin = {0, 0, i};
            wgpuQueueWriteTexture(m_device->queue(), &dest, white, sizeof(white), &layout, &extent);
        }
    }
    // Depth textures get their clear value (1.0) through render attachment clear

    WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    view_desc.format = format;
    view_desc.dimension = tex_view_dim;
    view_desc.mipLevelCount = 1;
    view_desc.arrayLayerCount = layers;
    auto view = wgpuTextureCreateView(texture, &view_desc);
    INVARIANT_MSG(view, "FallbackPool: failed to create fallback texture view");

    m_textures[key] = {texture, view};
    return view;
}

WGPUBuffer FallbackPool::buffer(uint64_t min_size) {
    // Find an existing buffer >= min_size
    for (auto& entry : m_buffers) {
        if (entry.size >= min_size) return entry.buffer;
    }

    // Create a new zero-filled buffer
    uint64_t size = std::max(min_size, uint64_t(256));  // WebGPU minimum
    WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    buf_desc.size = size;
    buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    auto buf = wgpuDeviceCreateBuffer(m_device->handle(), &buf_desc);
    INVARIANT_MSG(buf, "FallbackPool: failed to create fallback buffer");

    m_buffers.push_back({buf, size});
    return buf;
}

}  // namespace pts::rendering
