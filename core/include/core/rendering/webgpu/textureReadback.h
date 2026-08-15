#pragma once

#include <core/rendering/webgpu/asyncStateMachine.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace pts::webgpu {

namespace texture_readback {
struct IdleState {};
struct CopyingState {};
struct MappedState {};
/// Terminal: buffer mapping failed (typically because the device was lost).
/// Distinct from Idle so callers cannot mistake a failure for "nothing in
/// flight" and silently re-issue the readback forever.
struct FailedState {};
}  // namespace texture_readback

/// Async full-texture readback from GPU to CPU.
/// Usage:
///   1. Call request() -- records CopyTextureToBuffer on the encoder
///   2. Caller submits the encoder
///   3. Call tick() each frame -- drives mapAsync
///   4. Call try_read() -- returns pixel data when ready
class TextureReadback
    : public AsyncStateMachine<TextureReadback, texture_readback::IdleState,
                               texture_readback::CopyingState, texture_readback::MappedState,
                               texture_readback::FailedState> {
   public:
    using AsyncStateMachine::tick;

    TextureReadback() = default;
    ~TextureReadback();

    TextureReadback(const TextureReadback&) = delete;
    TextureReadback& operator=(const TextureReadback&) = delete;
    TextureReadback(TextureReadback&&) noexcept;
    TextureReadback& operator=(TextureReadback&&) noexcept;

    /// Issue a full-texture readback. Records CopyTextureToBuffer on the encoder.
    /// The caller must submit the encoder, then call tick() each frame.
    void request(WGPUCommandEncoder encoder, WGPUTexture texture, uint32_t width, uint32_t height,
                 WGPUDevice device, WGPUInstance instance);

    /// Try to read the result. Returns contiguous RGBA8 pixels (width*height*4 bytes)
    /// with row padding stripped. Returns empty vector if not ready yet.
    /// After successful read, unmaps and transitions to Idle.
    [[nodiscard]] auto try_read() -> std::vector<uint8_t>;

    /// True if a readback is in flight (copy issued or mapping).
    [[nodiscard]] bool is_pending() const;

    /// True if the readback failed and will never produce pixels. Terminal --
    /// the readback must not be re-issued against the same (dead) device.
    [[nodiscard]] bool has_failed() const;

    // CRTP interface
    void on_tick();
    [[nodiscard]] WGPUInstance wgpu_instance() const;

   private:
    WGPUBuffer m_buffer = nullptr;
    WGPUInstance m_instance = nullptr;
    WGPUDevice m_device = nullptr;
    bool m_needs_map = false;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_bytes_per_row = 0;

    void cleanup();
};

}  // namespace pts::webgpu
