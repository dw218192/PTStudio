#pragma once

#include <core/diagnostics.h>
#include <core/rendering/webgpu/webgpu.h>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace pts::webgpu {
class Device;
}

namespace pts::rendering {

/// Describes a single binding slot in a bind group layout.
/// Each OutputSlot maps to exactly one WGPUBindGroupLayoutEntry.
struct OutputSlot {
    enum class Kind : uint8_t {
        Texture,         ///< Sampled texture
        Sampler,         ///< Sampler
        Uniform,         ///< Uniform buffer
        Storage,         ///< Storage buffer (read-only by default)
        StorageTexture,  ///< Write-only storage texture
    };

    Kind kind = Kind::Texture;
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    WGPUTextureViewDimension dimension = WGPUTextureViewDimension_2D;
    uint64_t min_buffer_size = 0;
    WGPUShaderStage vis = WGPUShaderStage_Fragment;
    WGPUSamplerBindingType sampler_type = WGPUSamplerBindingType_Filtering;
    WGPUAddressMode address_mode = WGPUAddressMode_ClampToEdge;
    WGPUMipmapFilterMode mipmap_filter = WGPUMipmapFilterMode_Nearest;
    bool has_dynamic_offset = false;
    bool is_read_write = false;

    // --- Chainable modifiers ---
    OutputSlot& dynamic() {
        has_dynamic_offset = true;
        return *this;
    }
    OutputSlot& read_write() {
        is_read_write = true;
        return *this;
    }
    OutputSlot& visibility(WGPUShaderStage stage) {
        vis = stage;
        return *this;
    }

    // --- Static factories ---

    /// Sampled texture (1 binding). Sample type derived from format.
    static OutputSlot texture(WGPUTextureFormat fmt,
                              WGPUTextureViewDimension dim = WGPUTextureViewDimension_2D) {
        OutputSlot s{};
        s.kind = Kind::Texture;
        s.format = fmt;
        s.dimension = dim;
        return s;
    }

    /// Uniform buffer (1 binding).
    static OutputSlot uniform(uint64_t min_size) {
        OutputSlot s{};
        s.kind = Kind::Uniform;
        s.min_buffer_size = min_size;
        return s;
    }

    /// Read-only storage buffer (1 binding). Use .read_write() for Storage.
    static OutputSlot storage(uint64_t min_size = 0) {
        OutputSlot s{};
        s.kind = Kind::Storage;
        s.min_buffer_size = min_size;
        return s;
    }

    /// Sampler (1 binding). Type specifies Filtering or NonFiltering.
    static OutputSlot sampler(WGPUSamplerBindingType type,
                              WGPUAddressMode address = WGPUAddressMode_ClampToEdge,
                              WGPUMipmapFilterMode mipmap = WGPUMipmapFilterMode_Nearest) {
        OutputSlot s{};
        s.kind = Kind::Sampler;
        s.sampler_type = type;
        s.address_mode = address;
        s.mipmap_filter = mipmap;
        return s;
    }

    /// Write-only storage texture (1 binding).
    static OutputSlot storage_texture(WGPUTextureFormat fmt,
                                      WGPUTextureViewDimension dim = WGPUTextureViewDimension_2D) {
        OutputSlot s{};
        s.kind = Kind::StorageTexture;
        s.format = fmt;
        s.dimension = dim;
        return s;
    }

    /// Convenience: sampled texture + paired sampler (2 slots).
    /// Sampler type auto-derived: depth → NonFiltering, else Filtering.
    static std::array<OutputSlot, 2> sampled_texture(
        WGPUTextureFormat fmt, WGPUTextureViewDimension dim = WGPUTextureViewDimension_2D);
};

/// Create a bind group layout from a flat list of OutputSlots.
/// Each slot = one binding, indices sequential starting at 0.
WGPUBindGroupLayout create_bind_group_layout(const webgpu::Device& device,
                                             std::initializer_list<OutputSlot> slots);

/// Overload accepting a vector (for concatenation from multiple sources).
WGPUBindGroupLayout create_bind_group_layout(const webgpu::Device& device,
                                             const std::vector<OutputSlot>& slots);

}  // namespace pts::rendering
