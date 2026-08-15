#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace pts::rendering {

/// Centralizes naming and dedup for persistent frame-graph textures used to
/// carry data across frames (temporal accumulation, history buffers, etc.).
///
/// Each call to request_persistent() vends a TextureDeclHandle backed by a
/// `Lifetime::Persistent` texture in the frame graph. Repeated calls with the
/// same logical name return the same handle so multiple passes can share the
/// slot without colliding on names.
///
/// The manager itself owns no GPU resources -- the FG holds them. The manager
/// just remembers which logical names have been bound to which FG labels so
/// callers don't need to coordinate naming.
class TemporalStorageManager {
   public:
    TemporalStorageManager() = default;
    ~TemporalStorageManager() = default;

    TemporalStorageManager(const TemporalStorageManager&) = delete;
    TemporalStorageManager& operator=(const TemporalStorageManager&) = delete;
    TemporalStorageManager(TemporalStorageManager&&) = delete;
    TemporalStorageManager& operator=(TemporalStorageManager&&) = delete;

    /// Get-or-create a persistent FG texture under the given logical name.
    /// On first call the texture is registered with the supplied size/format
    /// and `Lifetime::Persistent`; later calls reuse it (a resize is forwarded
    /// to FrameGraph::resize so the FG can recreate if dimensions change).
    /// `usage` is OR'd into the texture's accumulated usage flags so callers
    /// that want to render to or sample from the texture get the right flags.
    [[nodiscard]] TextureDeclHandle request_persistent(FrameGraph& fg, std::string_view name,
                                                       uint32_t width, uint32_t height,
                                                       WGPUTextureFormat format,
                                                       WGPUTextureUsage usage);

    /// Convenience wrapper around request_persistent() for ping-pong history
    /// buffers. Returns {read_handle, write_handle} keyed off `frame_index`
    /// parity: the two underlying persistent textures alternate roles each
    /// frame, so the "read" handle holds the value written on the previous
    /// invocation.
    struct PingPong {
        TextureDeclHandle read;   ///< previous-frame data
        TextureDeclHandle write;  ///< current-frame target
    };
    [[nodiscard]] PingPong request_ping_pong(FrameGraph& fg, std::string_view base_name,
                                             uint32_t width, uint32_t height,
                                             WGPUTextureFormat format, WGPUTextureUsage usage,
                                             uint64_t frame_index);

   private:
    struct Entry {
        std::string fg_label;
        TextureDeclHandle handle;
        uint32_t width = 0;
        uint32_t height = 0;
        WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    };
    std::unordered_map<std::string, Entry> m_entries;
};

}  // namespace pts::rendering
