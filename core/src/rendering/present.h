#pragma once

#include <core/rendering/windowing.h>

#include <cstdint>
#include <memory>

#include "rhi.h"

namespace pts::rendering {

enum class PresentStatus : std::uint32_t {
    ok = 0,
    suboptimal = 1,
    out_of_date = 2,
    error = 3,
};

class IPresent {
   public:
    virtual ~IPresent() = default;

    [[nodiscard]] virtual auto acquire_next_backbuffer(RhiSemaphore signal_semaphore,
                                                       uint32_t* index) -> PresentStatus = 0;
    [[nodiscard]] virtual auto present_backbuffer(uint32_t index, RhiSemaphore wait_semaphore)
        -> PresentStatus = 0;
    virtual void recreate_swapchain() = 0;
    [[nodiscard]] virtual auto framebuffer_extent() const noexcept -> FramebufferExtent = 0;
};
}  // namespace pts::rendering
