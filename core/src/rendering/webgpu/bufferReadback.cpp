#include <core/diagnostics.h>
#include <core/rendering/webgpu/bufferReadback.h>

#include <algorithm>
#include <cstring>

namespace pts::webgpu {

// -- lifecycle ----------------------------------------------------------------

BufferReadback::~BufferReadback() {
    if (m_buffer) {
        if (is<MappedState>()) {
            wgpuBufferUnmap(m_buffer);
        }
        wgpuBufferRelease(m_buffer);
    }
}

BufferReadback::BufferReadback(BufferReadback&& other) noexcept
    : AsyncStateMachine(std::move(other)),
      m_buffer(other.m_buffer),
      m_instance(other.m_instance),
      m_device(other.m_device),
      m_needs_map(other.m_needs_map),
      m_region_size(other.m_region_size),
      m_buffer_size(other.m_buffer_size) {
    other.m_buffer = nullptr;
    other.m_instance = nullptr;
    other.m_device = nullptr;
    other.m_needs_map = false;
}

BufferReadback& BufferReadback::operator=(BufferReadback&& other) noexcept {
    if (this != &other) {
        if (m_buffer) {
            if (is<MappedState>()) {
                wgpuBufferUnmap(m_buffer);
            }
            wgpuBufferRelease(m_buffer);
        }

        AsyncStateMachine::operator=(std::move(other));
        m_buffer = other.m_buffer;
        m_instance = other.m_instance;
        m_device = other.m_device;
        m_needs_map = other.m_needs_map;
        m_region_size = other.m_region_size;
        m_buffer_size = other.m_buffer_size;
        other.m_buffer = nullptr;
        other.m_instance = nullptr;
        other.m_device = nullptr;
        other.m_needs_map = false;
    }
    return *this;
}

// WebGPU requires bytesPerRow to be a multiple of 256
static constexpr uint32_t k_bytes_per_row = 256;

// -- public API ---------------------------------------------------------------

void BufferReadback::request(WGPUCommandEncoder encoder, WGPUTexture texture, uint32_t x,
                             uint32_t y, WGPUDevice device, WGPUInstance instance,
                             uint32_t radius) {
    PRECONDITION(encoder);
    PRECONDITION(texture);
    PRECONDITION(device);
    PRECONDITION(instance);
    PRECONDITION(is<IdleState>());

    m_device = device;
    m_instance = instance;

    uint32_t side = 2 * radius + 1;
    m_region_size = side;

    // Clamp origin so the region stays within texture bounds
    uint32_t ox = (x >= radius) ? x - radius : 0;
    uint32_t oy = (y >= radius) ? y - radius : 0;

    uint32_t required = k_bytes_per_row * side;
    ensure_buffer(required);

    WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    src.texture = texture;
    src.mipLevel = 0;
    src.origin = {ox, oy, 0};

    WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    dst.buffer = m_buffer;
    dst.layout.offset = 0;
    dst.layout.bytesPerRow = k_bytes_per_row;
    dst.layout.rowsPerImage = side;

    WGPUExtent3D extent = {side, side, 1};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);

    m_needs_map = true;
    transition<PendingState>();
}

auto BufferReadback::try_read_u32() -> std::optional<uint32_t> {
    if (!is<MappedState>()) {
        return std::nullopt;
    }

    auto* base =
        static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(m_buffer, 0, m_buffer_size));
    INVARIANT(base);

    // Scan region for the closest non-background pixel to the center.
    // Spiral outward from center: distance 0 first, then 1, etc.
    uint32_t center = m_region_size / 2;
    uint32_t best_id = UINT32_MAX;
    uint32_t best_dist2 = UINT32_MAX;

    for (uint32_t row = 0; row < m_region_size; ++row) {
        auto* row_data = reinterpret_cast<const uint32_t*>(base + row * k_bytes_per_row);
        for (uint32_t col = 0; col < m_region_size; ++col) {
            uint32_t id = row_data[col];
            if (id == UINT32_MAX) continue;
            uint32_t dx = (col > center) ? col - center : center - col;
            uint32_t dy = (row > center) ? row - center : center - row;
            uint32_t dist2 = dx * dx + dy * dy;
            if (dist2 < best_dist2) {
                best_dist2 = dist2;
                best_id = id;
            }
        }
    }

    wgpuBufferUnmap(m_buffer);
    transition<IdleState>();
    return best_id;
}

// -- CRTP interface -----------------------------------------------------------

void BufferReadback::on_tick() {
    if (!is<PendingState>()) return;

    if (m_needs_map) {
        WGPUBufferMapCallbackInfo map_cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        map_cb.mode = WGPUCallbackMode_AllowProcessEvents;
        map_cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* self_ptr, void*) {
            auto* self = static_cast<BufferReadback*>(self_ptr);
            if (status != WGPUMapAsyncStatus_Success) {
                self->transition<IdleState>();
            }
        };
        map_cb.userdata1 = this;
        wgpuBufferMapAsync(m_buffer, WGPUMapMode_Read, 0, m_buffer_size, map_cb);
        m_needs_map = false;
        return;
    }

    auto map_state = wgpuBufferGetMapState(m_buffer);
    if (map_state == WGPUBufferMapState_Mapped) {
        transition<MappedState>();
    }
}

bool BufferReadback::is_pending() const {
    return is<PendingState>();
}

WGPUInstance BufferReadback::wgpu_instance() const {
    return m_instance;
}

// -- private ------------------------------------------------------------------

void BufferReadback::ensure_buffer(uint32_t required_size) {
    if (m_buffer && m_buffer_size >= required_size) return;

    if (m_buffer) {
        wgpuBufferRelease(m_buffer);
        m_buffer = nullptr;
    }

    INVARIANT(m_device);

    m_buffer_size = std::max(required_size, uint32_t{256});

    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.size = m_buffer_size;
    desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    desc.mappedAtCreation = false;

    m_buffer = wgpuDeviceCreateBuffer(m_device, &desc);
    INVARIANT(m_buffer);
}

}  // namespace pts::webgpu
