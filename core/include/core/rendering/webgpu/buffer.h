#pragma once

#include <core/rendering/webgpu/webgpu.h>

#include <cstddef>

namespace pts::webgpu {

class Buffer {
   public:
    Buffer() = default;
    explicit Buffer(WGPUBuffer buffer, std::size_t size);

    Buffer(const Buffer&) = delete;
    auto operator=(const Buffer&) -> Buffer& = delete;

    Buffer(Buffer&& other) noexcept;
    auto operator=(Buffer&& other) noexcept -> Buffer&;

    ~Buffer();

    [[nodiscard]] auto is_valid() const noexcept -> bool;
    [[nodiscard]] auto handle() const noexcept -> WGPUBuffer;
    [[nodiscard]] auto size() const noexcept -> std::size_t;

   private:
    WGPUBuffer m_buffer = nullptr;
    std::size_t m_size = 0;
};

}  // namespace pts::webgpu
