#include <core/rendering/scenePass.h>
#include <core/rendering/webgpu/device.h>
#include <fmt/format.h>

namespace pts::rendering {

void IScenePass::validate_debug_limits(const webgpu::Device& device) {
    auto [names, count] = debug_target_names();
    if (count == 0) return;
    uint32_t total_targets = 1 + count;  // scene_color + debug targets
    WGPULimits limits = WGPU_LIMITS_INIT;
    wgpuDeviceGetLimits(device.handle(), &limits);
    INVARIANT_MSG(total_targets <= limits.maxColorAttachments,
                  fmt::format("pass '{}' needs {} color attachments but device supports {}", name(),
                              total_targets, limits.maxColorAttachments)
                      .c_str());
    uint32_t bytes_per_sample = total_targets * 8;  // RGBA8Unorm aligned to 8
    INVARIANT_MSG(bytes_per_sample <= limits.maxColorAttachmentBytesPerSample,
                  fmt::format("pass '{}' needs {} bytes/sample but device supports {}", name(),
                              bytes_per_sample, limits.maxColorAttachmentBytesPerSample)
                      .c_str());
}

}  // namespace pts::rendering
