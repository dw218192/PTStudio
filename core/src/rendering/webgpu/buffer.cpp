#include <core/rendering/webgpu/buffer.h>

namespace pts::webgpu {

Buffer::Buffer(WGPUBuffer buffer, std::size_t size) : m_buffer(buffer), m_size(size) {
}

Buffer::Buffer(Buffer&& other) noexcept : m_buffer(other.m_buffer), m_size(other.m_size) {
    other.m_buffer = nullptr;
    other.m_size = 0;
}

auto Buffer::operator=(Buffer&& other) noexcept -> Buffer& {
    if (this != &other) {
        if (m_buffer != nullptr) {
            wgpuBufferRelease(m_buffer);
        }
        m_buffer = other.m_buffer;
        m_size = other.m_size;
        other.m_buffer = nullptr;
        other.m_size = 0;
    }
    return *this;
}

Buffer::~Buffer() {
    if (m_buffer != nullptr) {
        wgpuBufferRelease(m_buffer);
    }
}

auto Buffer::is_valid() const noexcept -> bool {
    return m_buffer != nullptr;
}

auto Buffer::handle() const noexcept -> WGPUBuffer {
    return m_buffer;
}

auto Buffer::size() const noexcept -> std::size_t {
    return m_size;
}

}  // namespace pts::webgpu
