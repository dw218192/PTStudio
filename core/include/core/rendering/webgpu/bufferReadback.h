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

    /// Issue a readback: records CopyTextureToBuffer on the given encoder.
    /// The caller must submit the encoder, then call tick() to begin the async map.
    void request(WGPUCommandEncoder encoder, WGPUTexture texture, uint32_t x, uint32_t y,
                 WGPUDevice device, WGPUInstance instance);

    /// Try to read the result. Returns the u32 value if mapped, nullopt otherwise.
    /// After successful read, unmaps and transitions to Idle.
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

    void ensure_buffer();
};

}  // namespace pts::webgpu
