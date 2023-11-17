#include "vulkanHelper.h"

[[nodiscard]] auto PTS::do_work_now(
    VulkanDeviceInfo const& dev,
    VulkanCmdPoolInfo const& cmd_pool,
    std::function<void(vk::CommandBuffer&)> work
) -> tl::expected<void, std::string>
{
    try {
        auto cmd_bufs = dev->allocateCommandBuffersUnique(
            vk::CommandBufferAllocateInfo{
                *cmd_pool,
                vk::CommandBufferLevel::ePrimary,
                1
            }
        );
        auto& cmd_buf = *cmd_bufs[0];
        cmd_buf.begin(vk::CommandBufferBeginInfo{});
        work(cmd_buf);
        cmd_buf.end();

        auto const fence = dev->createFenceUnique(vk::FenceCreateInfo{});
        cmd_pool.queue.submit(
            vk::SubmitInfo{}
                .setCommandBufferCount(1)
                .setPCommandBuffers(&cmd_buf),
            *fence
        );

        auto const res = dev->waitForFences(*fence, true, std::numeric_limits<uint64_t>::max());
        if (res != vk::Result::eSuccess) {
            return TL_ERROR("failed to wait for fence");
        }

        dev->resetFences(*fence);
        return {};
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}