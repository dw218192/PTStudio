#include <core/diagnostics.h>
#include <core/rendering/fallbackPool.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/outputLayout.h>
#include <core/rendering/renderPass.h>
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

void OutputLayoutInfo::release() {
    for (auto& slot : slots) {
        if (slot.sampler) {
            wgpuSamplerRelease(slot.sampler);
            slot.sampler = nullptr;
        }
    }
    if (layout) {
        wgpuBindGroupLayoutRelease(layout);
        layout = nullptr;
    }
}

std::vector<OutputSlot> OutputLayoutInfo::output_slots() const {
    std::vector<OutputSlot> out;
    out.reserve(slots.size());
    for (auto& si : slots) {
        out.push_back(si.slot);
    }
    return out;
}

static WGPUSampler create_sampler_for_slot(const webgpu::Device& device, const OutputSlot& slot) {
    PRECONDITION(slot.kind == OutputSlot::Kind::Sampler);
    WGPUSamplerDescriptor desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    desc.addressModeU = slot.address_mode;
    desc.addressModeV = slot.address_mode;
    desc.addressModeW = slot.address_mode;

    if (slot.sampler_type == WGPUSamplerBindingType_Filtering) {
        desc.magFilter = WGPUFilterMode_Linear;
        desc.minFilter = WGPUFilterMode_Linear;
    } else {
        desc.magFilter = WGPUFilterMode_Nearest;
        desc.minFilter = WGPUFilterMode_Nearest;
    }
    desc.mipmapFilter = slot.mipmap_filter;

    auto sampler = wgpuDeviceCreateSampler(device.handle(), &desc);
    INVARIANT_MSG(sampler, "create_output_layout: failed to create sampler");
    return sampler;
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

static OutputLayoutInfo create_output_layout_impl(const webgpu::Device& device,
                                                  const OutputSlot* slot_data, size_t slot_count) {
    OutputLayoutInfo info;
    info.slots.reserve(slot_count);

    std::vector<WGPUBindGroupLayoutEntry> entries;
    entries.reserve(slot_count);

    uint32_t binding = 0;
    for (size_t i = 0; i < slot_count; ++i) {
        auto& slot = slot_data[i];
        OutputLayoutInfo::SlotInfo si{};
        si.slot = slot;
        si.binding = binding;

        entries.push_back(make_bgl_entry(slot, binding));
        ++binding;

        if (slot.kind == OutputSlot::Kind::Sampler) {
            si.sampler = create_sampler_for_slot(device, slot);
        }

        info.slots.push_back(si);
    }

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = entries.size();
    bgl_desc.entries = entries.data();
    info.layout = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);
    INVARIANT_MSG(info.layout, "create_output_layout: failed to create bind group layout");

    return info;
}

OutputLayoutInfo create_output_layout(const webgpu::Device& device,
                                      std::initializer_list<OutputSlot> slots) {
    return create_output_layout_impl(device, slots.begin(), slots.size());
}

OutputLayoutInfo create_output_layout(const webgpu::Device& device,
                                      const std::vector<OutputSlot>& slots) {
    return create_output_layout_impl(device, slots.data(), slots.size());
}

// --- OutputLayoutInfo::build() ---

static DescriptorHandle build_impl(const OutputLayoutInfo& info, FrameGraph& fg, const IPass* pass,
                                   const BuildResource* res_data, size_t res_count,
                                   FallbackPool& pool, const char* label) {
    // Count non-sampler slots to validate resource count
    size_t non_sampler_count = 0;
    for (auto& si : info.slots) {
        if (si.slot.kind != OutputSlot::Kind::Sampler) ++non_sampler_count;
    }
    INVARIANT_MSG(res_count == non_sampler_count,
                  "build: resource count must match non-sampler slot count");

    auto builder = fg.descriptor(pass, info.layout, label);

    size_t res_index = 0;
    for (auto& si : info.slots) {
        uint32_t b = si.binding;

        if (si.slot.kind == OutputSlot::Kind::Sampler) {
            INVARIANT_MSG(si.sampler, "build: sampler slot missing pre-created sampler");
            builder.sampler(b, si.sampler);
            continue;
        }

        INVARIANT(res_index < res_count);
        auto& resource = res_data[res_index++];

        switch (si.slot.kind) {
            case OutputSlot::Kind::Texture: {
                if (auto* tex = std::get_if<TextureHandle>(&resource)) {
                    if (tex->is_valid()) {
                        builder.texture(b, *tex);
                    } else {
                        auto fallback_view = pool.view(si.slot.format, si.slot.dimension);
                        builder.external_view(b, fallback_view);
                    }
                } else if (auto* view = std::get_if<WGPUTextureView>(&resource)) {
                    builder.external_view(b, *view);
                } else {
                    PANIC("build: texture slot requires TextureHandle or WGPUTextureView");
                }
                break;
            }

            case OutputSlot::Kind::Uniform:
            case OutputSlot::Kind::Storage: {
                if (auto* buf = std::get_if<BufferHandle>(&resource)) {
                    builder.buffer(b, *buf, 0, si.slot.min_buffer_size);
                } else if (auto* raw_buf = std::get_if<WGPUBuffer>(&resource)) {
                    builder.external_buffer(b, *raw_buf, 0, si.slot.min_buffer_size);
                } else {
                    PANIC("build: buffer slot requires BufferHandle or WGPUBuffer");
                }
                break;
            }

            case OutputSlot::Kind::StorageTexture: {
                if (auto* tex = std::get_if<TextureHandle>(&resource)) {
                    builder.texture(b, *tex);
                } else if (auto* view = std::get_if<WGPUTextureView>(&resource)) {
                    builder.external_view(b, *view);
                } else {
                    PANIC("build: storage texture slot requires TextureHandle or WGPUTextureView");
                }
                break;
            }

            case OutputSlot::Kind::Sampler:
                UNREACHABLE();
        }
    }

    return builder.build();
}

DescriptorHandle OutputLayoutInfo::build(FrameGraph& fg, const IPass* pass,
                                         std::initializer_list<BuildResource> resources,
                                         FallbackPool& pool, const char* label) const {
    return build_impl(*this, fg, pass, resources.begin(), resources.size(), pool, label);
}

DescriptorHandle OutputLayoutInfo::build(FrameGraph& fg, const IPass* pass,
                                         const std::vector<BuildResource>& resources,
                                         FallbackPool& pool, const char* label) const {
    return build_impl(*this, fg, pass, resources.data(), resources.size(), pool, label);
}

}  // namespace pts::rendering
