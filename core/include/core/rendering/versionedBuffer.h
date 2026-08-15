#pragma once

#include <core/diagnostics.h>
#include <core/rendering/webgpu/webgpu.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace pts::rendering {

/// Value type handed from producers (e.g. RenderWorld) to the FrameGraph.
/// Collapses the (handle, size_bytes, version) triple into a single
/// copy-cheap scalar bundle. The underlying WGPUBuffer is owned elsewhere;
/// this is a non-owning view that FrameGraph uses to key descriptor cache
/// invalidation.
struct ImportedBuffer {
    WGPUBuffer handle = nullptr;
    uint64_t size_bytes = 0;
    uint64_t version = 0;
};

/// Owning, move-only GPU buffer wrapper that auto-bumps a monotonic version
/// on every write. Callers never touch the version directly -- it changes as
/// a side-effect of write/write_at, so the (handle, size, version) triple
/// handed to the FrameGraph is always consistent with the buffer contents.
///
/// Growth policy: write(count) reallocates to max(1, count) * sizeof(T)
/// bytes if the current capacity is smaller. The old WGPUBuffer handle is
/// released synchronously; bind groups referencing it keep it alive via
/// Dawn's internal reference counting.
template <typename T>
class VersionedBuffer {
   public:
    VersionedBuffer() = default;

    VersionedBuffer(WGPUDevice device, uint64_t initial_count, WGPUBufferUsage usage,
                    const char* label = nullptr)
        : m_device(device), m_usage(usage) {
        PRECONDITION_MSG(device != nullptr, "VersionedBuffer: device must not be null");
        allocate(std::max<uint64_t>(1, initial_count), label);
    }

    ~VersionedBuffer() {
        release();
    }

    VersionedBuffer(const VersionedBuffer&) = delete;
    VersionedBuffer& operator=(const VersionedBuffer&) = delete;

    VersionedBuffer(VersionedBuffer&& other) noexcept
        : m_device(other.m_device),
          m_handle(other.m_handle),
          m_size_bytes(other.m_size_bytes),
          m_version(other.m_version),
          m_usage(other.m_usage) {
        other.m_handle = nullptr;
        other.m_size_bytes = 0;
        // Leave m_device / m_usage / m_version on the moved-from object -- the
        // only valid post-move operations are destruction or move-assign, and
        // version read is fine (reports the last state before move).
    }

    VersionedBuffer& operator=(VersionedBuffer&& other) noexcept {
        if (this != &other) {
            release();
            m_device = other.m_device;
            m_handle = other.m_handle;
            m_size_bytes = other.m_size_bytes;
            m_version = other.m_version;
            m_usage = other.m_usage;
            other.m_handle = nullptr;
            other.m_size_bytes = 0;
        }
        return *this;
    }

    /// Full upload starting at offset 0. Grows the buffer if
    /// `count * sizeof(T)` exceeds current capacity. Always bumps the
    /// version, even when count == 0 (the empty-write case still counts as
    /// a logical update). Skips the actual wgpuQueueWriteBuffer call when
    /// count == 0.
    void write(WGPUQueue queue, const T* data, std::size_t count) {
        PRECONDITION_MSG(m_device != nullptr, "VersionedBuffer::write on default-constructed");
        PRECONDITION_MSG(queue != nullptr, "VersionedBuffer::write: queue must not be null");
        auto required_bytes = static_cast<uint64_t>(count) * sizeof(T);
        if (required_bytes > m_size_bytes) {
            release();
            allocate(std::max<uint64_t>(1, count), nullptr);
        }
        if (count > 0) {
            PRECONDITION_MSG(data != nullptr,
                             "VersionedBuffer::write: data must not be null when count > 0");
            wgpuQueueWriteBuffer(queue, m_handle, 0, data, required_bytes);
        }
        ++m_version;
    }

    /// Partial upload at an offset measured in elements of T. Does not grow
    /// the buffer -- caller guarantees `(offset_elements + count) * sizeof(T)
    /// <= size_bytes()`. Bumps the version.
    void write_at(WGPUQueue queue, std::size_t offset_elements, const T* data, std::size_t count) {
        PRECONDITION_MSG(m_handle != nullptr, "VersionedBuffer::write_at on empty buffer");
        PRECONDITION_MSG(queue != nullptr, "VersionedBuffer::write_at: queue must not be null");
        auto end_bytes =
            static_cast<uint64_t>(offset_elements + count) * static_cast<uint64_t>(sizeof(T));
        PRECONDITION_MSG(end_bytes <= m_size_bytes,
                         "VersionedBuffer::write_at: write would exceed buffer capacity");
        if (count > 0) {
            PRECONDITION_MSG(data != nullptr,
                             "VersionedBuffer::write_at: data must not be null when count > 0");
            wgpuQueueWriteBuffer(queue, m_handle,
                                 static_cast<uint64_t>(offset_elements) * sizeof(T), data,
                                 static_cast<uint64_t>(count) * sizeof(T));
        }
        ++m_version;
    }

    /// Zero-arg handoff to the FrameGraph consumer. Returns a value that
    /// collapses the (handle, size, version) triple the FG needs to key its
    /// descriptor cache.
    [[nodiscard]] ImportedBuffer imported() const noexcept {
        return ImportedBuffer{m_handle, m_size_bytes, m_version};
    }

    [[nodiscard]] WGPUBuffer handle() const noexcept {
        return m_handle;
    }
    [[nodiscard]] uint64_t size_bytes() const noexcept {
        return m_size_bytes;
    }
    [[nodiscard]] uint64_t version() const noexcept {
        return m_version;
    }
    [[nodiscard]] bool is_valid() const noexcept {
        return m_handle != nullptr;
    }

   private:
    void allocate(uint64_t element_count, const char* label) {
        PRECONDITION_MSG(m_device != nullptr, "VersionedBuffer::allocate: device must be set");
        WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        desc.size = element_count * sizeof(T);
        desc.usage = m_usage;
        if (label != nullptr) {
            desc.label = {label, WGPU_STRLEN};
        }
        m_handle = wgpuDeviceCreateBuffer(m_device, &desc);
        INVARIANT_MSG(m_handle != nullptr, "VersionedBuffer: wgpuDeviceCreateBuffer failed");
        m_size_bytes = desc.size;
    }

    void release() {
        if (m_handle != nullptr) {
            wgpuBufferRelease(m_handle);
            m_handle = nullptr;
        }
        m_size_bytes = 0;
    }

    WGPUDevice m_device = nullptr;
    WGPUBuffer m_handle = nullptr;
    uint64_t m_size_bytes = 0;
    uint64_t m_version = 0;
    WGPUBufferUsage m_usage = WGPUBufferUsage_None;
};

}  // namespace pts::rendering
