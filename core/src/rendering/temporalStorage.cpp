#include <core/diagnostics.h>
#include <core/rendering/temporalStorage.h>

#include <string>
#include <utility>

namespace pts::rendering {

TextureDeclHandle TemporalStorageManager::request_persistent(FrameGraph& fg, std::string_view name,
                                                             uint32_t width, uint32_t height,
                                                             WGPUTextureFormat format,
                                                             WGPUTextureUsage usage) {
    PRECONDITION_MSG(!name.empty(), "TemporalStorageManager: name must not be empty");
    PRECONDITION_MSG(width > 0 && height > 0, "TemporalStorageManager: zero-sized texture");

    auto key = std::string(name);
    auto it = m_entries.find(key);
    if (it == m_entries.end()) {
        Entry entry;
        entry.fg_label = "temporal/" + key;
        TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = format;
        desc.usage = usage;
        entry.handle = fg.texture(entry.fg_label, desc, Lifetime::Persistent);
        entry.width = width;
        entry.height = height;
        entry.format = format;
        it = m_entries.emplace(std::move(key), std::move(entry)).first;
        return it->second.handle;
    }

    auto& entry = it->second;
    INVARIANT_MSG(entry.format == format,
                  "TemporalStorageManager: format mismatch on repeat request -- "
                  "callers must use a stable format per logical name");
    if (entry.width != width || entry.height != height) {
        TextureDesc new_desc;
        new_desc.width = width;
        new_desc.height = height;
        new_desc.format = format;
        new_desc.usage = usage;
        fg.resize(entry.handle, new_desc);
        entry.width = width;
        entry.height = height;
    } else {
        // Re-declare so the FG marks the slot active this frame and accumulates
        // any new usage flags from this caller.
        TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = format;
        desc.usage = usage;
        fg.texture(entry.fg_label, desc, Lifetime::Persistent);
    }
    return entry.handle;
}

TemporalStorageManager::PingPong TemporalStorageManager::request_ping_pong(
    FrameGraph& fg, std::string_view base_name, uint32_t width, uint32_t height,
    WGPUTextureFormat format, WGPUTextureUsage usage, uint64_t frame_index) {
    PRECONDITION_MSG(!base_name.empty(), "TemporalStorageManager: base_name must not be empty");

    auto name_a = std::string(base_name) + "_a";
    auto name_b = std::string(base_name) + "_b";
    auto handle_a = request_persistent(fg, name_a, width, height, format, usage);
    auto handle_b = request_persistent(fg, name_b, width, height, format, usage);

    PingPong out;
    if ((frame_index & 1u) == 0u) {
        out.read = handle_b;
        out.write = handle_a;
    } else {
        out.read = handle_a;
        out.write = handle_b;
    }
    return out;
}

}  // namespace pts::rendering
