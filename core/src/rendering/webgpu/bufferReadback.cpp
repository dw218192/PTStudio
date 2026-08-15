#include <core/diagnostics.h>
#include <core/rendering/webgpu/bufferReadback.h>

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
      m_needs_map(other.m_needs_map) {
    other.m_buffer = nullptr;
    other.m_instance = nullptr;
    other.m_device = nullptr;
    other.m_needs_map = false;
}

BufferReadback& BufferReadback::operator=(BufferReadback&& other) noexcept {
    if (this != &other) {
        // Clean up existing resources
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
        other.m_buffer = nullptr;
        other.m_instance = nullptr;
        other.m_device = nullptr;
        other.m_needs_map = false;
    }
    return *this;
}

// -- public API ---------------------------------------------------------------

void BufferReadback::request(WGPUCommandEncoder encoder, WGPUTexture texture, uint32_t x,
                             uint32_t y, WGPUDevice device, WGPUInstance instance) {
    PRECONDITION(encoder);
    PRECONDITION(texture);
    PRECONDITION(device);
    PRECONDITION(instance);
    PRECONDITION(is<IdleState>());

    m_device = device;
    m_instance = instance;

    ensure_buffer();

    // Copy single pixel from texture to readback buffer
    WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    src.texture = texture;
    src.mipLevel = 0;
    src.origin = {x, y, 0};

    WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    dst.buffer = m_buffer;
    dst.layout.offset = 0;
    dst.layout.bytesPerRow = 256;
    dst.layout.rowsPerImage = 1;

    WGPUExtent3D extent = {1, 1, 1};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);

    // Defer mapAsync to on_tick() -- the caller must submit the encoder first
    m_needs_map = true;
    transition<PendingState>();
}

auto BufferReadback::try_read_u32() -> std::optional<uint32_t> {
    if (!is<MappedState>()) {
        return std::nullopt;
    }

    auto* data =
        static_cast<const uint32_t*>(wgpuBufferGetConstMappedRange(m_buffer, 0, sizeof(uint32_t)));
    INVARIANT(data);

    uint32_t value = *data;
    wgpuBufferUnmap(m_buffer);
    transition<IdleState>();
    return value;
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
        wgpuBufferMapAsync(m_buffer, WGPUMapMode_Read, 0, 256, map_cb);
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

void BufferReadback::ensure_buffer() {
    if (m_buffer) return;

    INVARIANT(m_device);

    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.size = 256;
    desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    desc.mappedAtCreation = false;

    m_buffer = wgpuDeviceCreateBuffer(m_device, &desc);
    INVARIANT(m_buffer);
}

}  // namespace pts::webgpu
