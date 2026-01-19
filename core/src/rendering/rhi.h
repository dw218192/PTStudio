#pragma once

#include <core/rendering/cmd.h>

#include <cstdint>

namespace pts::rendering {

enum class RhiBackendType : std::uint32_t {
    vulkan = 1,
};

struct RhiDevice {
    void* handle{nullptr};
};

struct RhiQueue {
    void* handle{nullptr};
};

struct RhiFence {
    void* handle{nullptr};
};

struct RhiSemaphore {
    void* handle{nullptr};
};

class IRhi {
   public:
    virtual ~IRhi() = default;

    [[nodiscard]] virtual auto backend_type() const noexcept -> RhiBackendType = 0;
    [[nodiscard]] virtual auto device_handle() const noexcept -> RhiDevice = 0;
    [[nodiscard]] virtual auto graphics_queue_handle() const noexcept -> RhiQueue = 0;
    [[nodiscard]] virtual auto graphics_queue_family() const noexcept -> uint32_t = 0;
    [[nodiscard]] virtual auto cmd_api() const noexcept -> const PtsCmdApi* = 0;

    [[nodiscard]] virtual auto create_fence(bool signaled) -> RhiFence = 0;
    virtual void destroy_fence(RhiFence fence) noexcept = 0;
    virtual auto wait_fence(RhiFence fence, std::uint64_t timeout_ns) noexcept -> bool = 0;
    virtual void reset_fence(RhiFence fence) noexcept = 0;

    [[nodiscard]] virtual auto create_semaphore() -> RhiSemaphore = 0;
    virtual void destroy_semaphore(RhiSemaphore semaphore) noexcept = 0;
};
}  // namespace pts::rendering
