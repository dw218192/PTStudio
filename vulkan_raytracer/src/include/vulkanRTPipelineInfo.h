#pragma once
#include "params.h"
#include "shaderCommon.h"
#include "vulkanHelper.h"
#include "vulkanAccelStructInfo.h"
#include "vulkanBufferInfo.h"
#include "vulkanDescSetInfo.h"

namespace PTS {
	struct VulkanRTPipelineInfo : VulkanInfo<vk::UniquePipeline> {
		[[nodiscard]] static auto create(
			VulkanDeviceInfo const& dev,
			VulkanCmdPoolInfo const& cmd_pool,
			VulkanImageInfo const& output_img,
			VulkanDescSetPoolInfo const& desc_set_pool
		) -> tl::expected<VulkanRTPipelineInfo, std::string>;

		/**
		 * @brief bind the indices of some mesh to the index buffer at the given index
		 * @param dev the vulkan device
		 * @param index the index of the index buffer to bind to; usually this is the primitive index
		 * @param indices the indices to bind
		 * @return an error if the indices could not be bound
		*/
		[[nodiscard]] auto bind_indices(VulkanDeviceInfo const& dev, int index,
		                                tcb::span<unsigned const> indices) noexcept
			-> tl::expected<void, std::string>;

		/**
		 * @brief bind the vertex attributes of some mesh to the vertex buffer at the given index
		 * @param dev the vulkan device
		 * @param index the index of the vertex buffer to bind to; usually this is the primitive index
		 * @param vertices the vertex data to bind
		 * @return an error if the vertex attributes could not be bound
		 */
		[[nodiscard]] auto bind_vertex_attribs(VulkanDeviceInfo const& dev, int index,
		                                       tcb::span<Vertex const> vertices) noexcept
			-> tl::expected<void, std::string>;

		[[nodiscard]] auto get_desc_sets() {
			auto ret = std::array<vk::DescriptorSet, VulkanRayTracingShaders::RayTracingBindings::k_num_sets>{};
			std::transform(
				std::begin(desc_sets),
				std::end(desc_sets),
				std::begin(ret),
				[](auto const& desc_set) {
					return *desc_set;
				}
			);
			return ret;
		}

		vk::UniquePipelineLayout layout{};
		VulkanTopAccelStructInfo top_accel{};

		// Desc Set related
		std::array<VulkanDescSetInfo, VulkanRayTracingShaders::RayTracingBindings::k_num_sets> desc_sets{};

		// Uniforms related
		VulkanBufferInfo materials_mem{};
		VulkanBufferInfo lights_mem{};

		// SBT related
		VulkanBufferInfo raygen_mem{};
		VulkanBufferInfo miss_mem{};
		VulkanBufferInfo hit_mem{};
		vk::StridedDeviceAddressRegionKHR raygen_region{};
		vk::StridedDeviceAddressRegionKHR miss_region{};
		vk::StridedDeviceAddressRegionKHR hit_region{};

		VulkanBufferInfo m_null_buffer{};

		// Unbounded arrays
		std::array<VulkanBufferInfo, k_max_instances> vertex_attribs_ssbos{};
		std::array<VulkanBufferInfo, k_max_instances> index_ssbos{};
	};
} // namespace PTS
