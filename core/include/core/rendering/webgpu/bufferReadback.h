#pragma once

#include <core/rendering/webgpu/asyncStateMachine.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <optional>

namespace pts::webgpu {

struct IdleState {};
struct PendingState {};
struct MappedState {};

class BufferReadback
    : public AsyncStateMachine<BufferReadback, IdleState, PendingState, MappedState> {
   public:
    using AsyncStateMachine::tick;
    using AsyncStateMachine::tick_until_settled;

    BufferReadback() = default;
    ~BufferReadback();

    BufferReadback(const BufferReadback&) = delete;
    BufferReadback& operator=(const BufferReadback&) = delete;
    BufferReadback(BufferReadback&&) noexcept;
    BufferReadback& operator=(BufferReadback&&) noexcept;

    /// Issue a readback of a small region around (x, y). Copies a (2*radius+1)^2
    /// square from the R32Uint picking texture. Coordinates are clamped to texture bounds.
    void request(WGPUCommandEncoder encoder, WGPUTexture texture, uint32_t x, uint32_t y,
                 WGPUDevice device, WGPUInstance instance, uint32_t radius = 0);

    /// Try to read the result. Scans the readback region for the closest
    /// non-background (!=UINT32_MAX) pixel to the center. Returns nullopt if
    /// not yet mapped, UINT32_MAX if all pixels are background.
    [[nodiscard]] auto try_read_u32() -> std::optional<uint32_t>;

    // CRTP interface
    void on_tick();
    [[nodiscard]] bool is_pending() const;
    [[nodiscard]] WGPUInstance wgpu_instance() const;

   private:
    WGPUBuffer m_buffer = nullptr;
    WGPUInstance m_instance = nullptr;
    WGPUDevice m_device = nullptr;
    bool m_needs_map = false;
    uint32_t m_region_size = 1;    // side length of the readback region
    uint32_t m_buffer_size = 256;  // current buffer allocation size

    void ensure_buffer(uint32_t required_size);
};

}  // namespace pts::webgpu
