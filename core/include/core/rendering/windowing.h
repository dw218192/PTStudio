#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

namespace pts::rendering {

enum class WindowingType : std::uint32_t {
    glfw = 1,
};

struct WindowingHandle {
    WindowingType type;
    // handle specific to the windowing system
    void* window_handle;
    // native platform handle
    void* platform_handle;
};

struct FramebufferExtent {
    std::uint32_t width;
    std::uint32_t height;
};

struct WindowingVulkanExtensions {
    const char* const* names{nullptr};
    std::uint32_t count{0};
};

class IWindowing {
   public:
    virtual ~IWindowing() = default;

    [[nodiscard]] virtual auto native_handle() const noexcept -> WindowingHandle = 0;
    [[nodiscard]] virtual auto framebuffer_extent() const noexcept -> FramebufferExtent = 0;
    [[nodiscard]] virtual auto required_vulkan_instance_extensions() const noexcept
        -> WindowingVulkanExtensions = 0;
    virtual void wait_events() const = 0;
};

}  // namespace pts::rendering
