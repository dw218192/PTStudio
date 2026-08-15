#include <core/diagnostics.h>
#include <core/rendering/webgpu/textureReadback.h>

#include <cstring>

namespace pts::webgpu {

using namespace texture_readback;

// -- lifecycle ----------------------------------------------------------------

TextureReadback::~TextureReadback() {
    cleanup();
}

TextureReadback::TextureReadback(TextureReadback&& other) noexcept
    : AsyncStateMachine(std::move(other)),
      m_buffer(other.m_buffer),
      m_instance(other.m_instance),
      m_device(other.m_device),
      m_needs_map(other.m_needs_map),
      m_width(other.m_width),
      m_height(other.m_height),
      m_bytes_per_row(other.m_bytes_per_row) {
    other.m_buffer = nullptr;
    other.m_instance = nullptr;
    other.m_device = nullptr;
    other.m_needs_map = false;
}

TextureReadback& TextureReadback::operator=(TextureReadback&& other) noexcept {
    if (this != &other) {
        cleanup();
        AsyncStateMachine::operator=(std::move(other));
        m_buffer = other.m_buffer;
        m_instance = other.m_instance;
        m_device = other.m_device;
        m_needs_map = other.m_needs_map;
        m_width = other.m_width;
        m_height = other.m_height;
        m_bytes_per_row = other.m_bytes_per_row;
        other.m_buffer = nullptr;
        other.m_instance = nullptr;
        other.m_device = nullptr;
        other.m_needs_map = false;
    }
    return *this;
}

// -- public API ---------------------------------------------------------------

void TextureReadback::request(WGPUCommandEncoder encoder, WGPUTexture texture, uint32_t width,
                              uint32_t height, WGPUDevice device, WGPUInstance instance) {
    PRECONDITION(encoder);
    PRECONDITION(texture);
    PRECONDITION(device);
    PRECONDITION(instance);
    PRECONDITION(width > 0 && height > 0);
    PRECONDITION(is<IdleState>());

    m_device = device;
    m_instance = instance;
    m_width = width;
    m_height = height;

    // 256-byte row alignment required by WebGPU
    m_bytes_per_row = ((width * 4 + 255) / 256) * 256;
    uint32_t buf_size = m_bytes_per_row * height;

    // Create staging buffer
    WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    buf_desc.size = buf_size;
    buf_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    m_buffer = wgpuDeviceCreateBuffer(device, &buf_desc);
    INVARIANT(m_buffer);

    // Record copy command
    WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    src.texture = texture;
    src.mipLevel = 0;
    src.origin = {0, 0, 0};

    WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    dst.buffer = m_buffer;
    dst.layout.offset = 0;
    dst.layout.bytesPerRow = m_bytes_per_row;
    dst.layout.rowsPerImage = height;

    WGPUExtent3D extent = {width, height, 1};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);

    // Defer mapAsync to on_tick() -- caller must submit the encoder first
    m_needs_map = true;
    transition<CopyingState>();
}

auto TextureReadback::try_read() -> std::vector<uint8_t> {
    if (!is<MappedState>()) {
        return {};
    }

    uint32_t buf_size = m_bytes_per_row * m_height;
    auto const* mapped =
        static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(m_buffer, 0, buf_size));
    INVARIANT(mapped);

    // Copy row-by-row, stripping WebGPU row padding
    std::vector<uint8_t> pixels(m_width * m_height * 4);
    uint32_t const row_bytes = m_width * 4;
    for (uint32_t y = 0; y < m_height; ++y) {
        std::memcpy(&pixels[y * row_bytes], mapped + y * m_bytes_per_row, row_bytes);
    }

    wgpuBufferUnmap(m_buffer);
    wgpuBufferRelease(m_buffer);
    m_buffer = nullptr;
    transition<IdleState>();
    return pixels;
}

// -- CRTP interface -----------------------------------------------------------

void TextureReadback::on_tick() {
    if (!is<CopyingState>()) return;

    if (m_needs_map) {
        uint32_t buf_size = m_bytes_per_row * m_height;
        WGPUBufferMapCallbackInfo map_cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        map_cb.mode = WGPUCallbackMode_AllowProcessEvents;
        map_cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* self_ptr, void*) {
            auto* self = static_cast<TextureReadback*>(self_ptr);
            if (status != WGPUMapAsyncStatus_Success) {
                // Do NOT fall back to Idle. Callers gate re-requests on
                // is_pending(), so an Idle-on-failure reset reads as "nothing in
                // flight" and the caller re-issues the readback every frame --
                // an unbounded retry loop that never surfaces the error.
                wgpuBufferRelease(self->m_buffer);
                self->m_buffer = nullptr;
                self->transition<FailedState>();
            }
        };
        map_cb.userdata1 = this;
        wgpuBufferMapAsync(m_buffer, WGPUMapMode_Read, 0, buf_size, map_cb);
        m_needs_map = false;
        return;
    }

    auto map_state = wgpuBufferGetMapState(m_buffer);
    if (map_state == WGPUBufferMapState_Mapped) {
        transition<MappedState>();
    }
}

bool TextureReadback::is_pending() const {
    return is<CopyingState>();
}

bool TextureReadback::has_failed() const {
    return is<FailedState>();
}

WGPUInstance TextureReadback::wgpu_instance() const {
    return m_instance;
}

// -- private ------------------------------------------------------------------

void TextureReadback::cleanup() {
    if (m_buffer) {
        if (is<MappedState>()) {
            wgpuBufferUnmap(m_buffer);
        }
        wgpuBufferRelease(m_buffer);
        m_buffer = nullptr;
    }
}

}  // namespace pts::webgpu
