#pragma once
#include "vulkanHelper.h"
#include "vulkanAccelStructInfo.h"
#include "vulkanBufferInfo.h"

namespace PTS {
struct VulkanRTPipelineInfo : VulkanInfo<vk::UniquePipeline> {
    vk::UniquePipelineLayout layout{};
    VulkanTopAccelStructInfo top_accel{};

    // Desc Set related
    VulkanDescSetInfo desc_set{};

    // Uniforms related
    VulkanBufferInfo camera_mem{};
    VulkanBufferInfo materials_mem{};

    // SBT related
    VulkanBufferInfo raygen_mem{};
    VulkanBufferInfo miss_mem{};
    VulkanBufferInfo hit_mem{};
    vk::StridedDeviceAddressRegionKHR raygen_region{};
    vk::StridedDeviceAddressRegionKHR miss_region{};
    vk::StridedDeviceAddressRegionKHR hit_region{};
};
} // namespace PTS