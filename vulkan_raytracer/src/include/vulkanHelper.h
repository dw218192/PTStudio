#pragma once
#include <vector>
#include <vulkan/vulkan.hpp>
#include <tl/expected.hpp>
#include <glm/glm.hpp>
#include <string_view>
#include <string>
#include <memory>
#include <tcb/span.hpp>

#include "vulkanGLInterop.h"

//TODO: refactor this file
namespace PTS {
    /**
     * @brief A wrapper for vulkan unique handles; used to logically bundle a handle with other related data
     * @tparam UniqueHandle vulkan unique handle type
    */
    template<typename UniqueHandle>
    struct VulkanInfo {
        UniqueHandle handle;
        operator bool() const {
            return handle.get();
        }
        auto* operator->() {
            return handle.operator->();
        }
        auto& operator*() {
            return handle.operator*();
        }
        auto const* operator->() const {
            return handle.operator->();
        }
        auto const& operator*() const {
            return handle.operator*();
        }
        auto* get() const {
            return handle.get();
        }
        auto const* get() {
            return handle.get();
        }
    };
    struct VulkanInsInfo : VulkanInfo<vk::UniqueInstance> {
        std::vector<std::string_view> ins_exts {};
        std::vector<std::string_view> layers {};
    };
    struct VulkanDeviceInfo : VulkanInfo<vk::UniqueDevice> {
        vk::PhysicalDevice physical_device {};
        unsigned queue_family_idx{ 0 };
    };
    struct VulkanCmdPoolInfo : VulkanInfo<vk::UniqueCommandPool> {
        vk::Queue queue {};
    };
    struct VulkanDescSetPoolInfo : VulkanInfo<vk::UniqueDescriptorPool> {};

    // if not shared, only vulkan part of this struct is valid
    struct VulkanImageInfo {
        VulkanGLInteropUtils::SharedImage img{};
        vk::UniqueImageView view {};
        vk::ImageLayout layout {};
        vk::UniqueSampler sampler {};
        auto get_desc_info() const noexcept -> vk::DescriptorImageInfo {
            return vk::DescriptorImageInfo { *sampler, *view, layout };
        }
    };
    struct VulkanRenderPassInfo : VulkanInfo<vk::UniqueRenderPass> {
        vk::Format color_fmt {};
        vk::Format depth_fmt {};
    };
    struct VulkanFrameBufferInfo : VulkanInfo<vk::UniqueFramebuffer> {
        VulkanImageInfo color_tex {};
        VulkanImageInfo depth_tex {};
    };
    struct VulkanShaderInfo : VulkanInfo<vk::UniqueShaderModule> {
        vk::ShaderStageFlagBits stage {};
    };
    struct VulkanCmdBufInfo : VulkanInfo<vk::UniqueCommandBuffer> {};

    [[nodiscard]] auto do_work_now(
        VulkanDeviceInfo const& dev,
        VulkanCmdPoolInfo const& cmd_pool,
        std::function<void(vk::CommandBuffer&)> work
    ) -> tl::expected<void, std::string>;
}