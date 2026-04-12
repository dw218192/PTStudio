#include <core/diagnostics.h>
#include <core/rendering/outputLayout.h>
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

std::array<OutputSlot, 2> OutputSlot::sampled_texture(WGPUTextureFormat fmt,
                                                      WGPUTextureViewDimension dim) {
    bool depth = is_depth_format(fmt);
    return {
        OutputSlot::texture(fmt, dim),
        OutputSlot::sampler(depth ? WGPUSamplerBindingType_NonFiltering
                                  : WGPUSamplerBindingType_Filtering),
    };
}

static WGPUBindGroupLayoutEntry make_bgl_entry(const OutputSlot& slot, uint32_t binding) {
    WGPUBindGroupLayoutEntry e = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    e.binding = binding;
    e.visibility = slot.vis;

    switch (slot.kind) {
        case OutputSlot::Kind::Texture: {
            e.texture.sampleType = is_depth_format(slot.format)
                                       ? WGPUTextureSampleType_UnfilterableFloat
                                       : WGPUTextureSampleType_Float;
            e.texture.viewDimension = slot.dimension;
            break;
        }
        case OutputSlot::Kind::Sampler: {
            e.sampler.type = slot.sampler_type;
            break;
        }
        case OutputSlot::Kind::Uniform: {
            e.buffer.type = WGPUBufferBindingType_Uniform;
            e.buffer.hasDynamicOffset = slot.has_dynamic_offset;
            e.buffer.minBindingSize = slot.min_buffer_size;
            break;
        }
        case OutputSlot::Kind::Storage: {
            e.buffer.type = slot.is_read_write ? WGPUBufferBindingType_Storage
                                               : WGPUBufferBindingType_ReadOnlyStorage;
            e.buffer.minBindingSize = slot.min_buffer_size;
            break;
        }
        case OutputSlot::Kind::StorageTexture: {
            e.storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
            e.storageTexture.format = slot.format;
            e.storageTexture.viewDimension = slot.dimension;
            break;
        }
    }
    return e;
}

static WGPUBindGroupLayout create_bgl_impl(const webgpu::Device& device,
                                           const OutputSlot* slot_data, size_t slot_count) {
    std::vector<WGPUBindGroupLayoutEntry> entries;
    entries.reserve(slot_count);

    for (size_t i = 0; i < slot_count; ++i) {
        entries.push_back(make_bgl_entry(slot_data[i], static_cast<uint32_t>(i)));
    }

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = entries.size();
    bgl_desc.entries = entries.data();
    auto layout = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);
    INVARIANT_MSG(layout, "create_bind_group_layout: failed to create bind group layout");

    return layout;
}

WGPUBindGroupLayout create_bind_group_layout(const webgpu::Device& device,
                                             std::initializer_list<OutputSlot> slots) {
    return create_bgl_impl(device, slots.begin(), slots.size());
}

WGPUBindGroupLayout create_bind_group_layout(const webgpu::Device& device,
                                             const std::vector<OutputSlot>& slots) {
    return create_bgl_impl(device, slots.data(), slots.size());
}

}  // namespace pts::rendering
