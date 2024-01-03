#include "vulkanAccelStructInfo.h"
#include "params.h"
#include "shaderCommon.h"

#ifndef NDEBUG
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#endif

[[nodiscard]] auto PTS::VulkanAccelStructInfo::create(
	VulkanDeviceInfo const& dev,
	VulkanCmdPoolInfo const& cmd_pool,
	vk::AccelerationStructureBuildGeometryInfoKHR geom_build_info,
	uint32_t primitive_count,
	uint32_t max_primitive_count
) -> tl::expected<VulkanAccelStructInfo, std::string> {
	auto build_sizes = dev->getAccelerationStructureBuildSizesKHR(
		vk::AccelerationStructureBuildTypeKHR::eDevice,
		geom_build_info,
		max_primitive_count
	);
	auto accel_buf = VulkanBufferInfo{};
	TL_TRY_ASSIGN(accel_buf, VulkanBufferInfo::create(
		              dev,
		              VulkanBufferInfo::Type::AccelStorage,
		              build_sizes.accelerationStructureSize
	              ));
	auto accel = dev->createAccelerationStructureKHRUnique(
		vk::AccelerationStructureCreateInfoKHR{}
		.setBuffer(*accel_buf)
		.setSize(build_sizes.accelerationStructureSize)
		.setType(geom_build_info.type)
	);

	auto scratch_buf = VulkanBufferInfo{};
	TL_TRY_ASSIGN(scratch_buf, VulkanBufferInfo::create(
		              dev,
		              VulkanBufferInfo::Type::Scratch,
		              build_sizes.buildScratchSize
	              ));

	geom_build_info
		.setScratchData(scratch_buf.get_device_addr())
		.setDstAccelerationStructure(*accel);

	auto cmd_buf = vk::UniqueCommandBuffer{};
	try {
		auto cmd_bufs = dev->allocateCommandBuffersUnique(
			vk::CommandBufferAllocateInfo{
				*cmd_pool,
				vk::CommandBufferLevel::ePrimary,
				1
			}
		);
		cmd_buf = std::move(cmd_bufs[0]);
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
	auto build_range_info = vk::AccelerationStructureBuildRangeInfoKHR{}
	                        .setFirstVertex(0)
	                        .setPrimitiveCount(max_primitive_count)
	                        .setPrimitiveOffset(0)
	                        .setTransformOffset(0);

	TL_CHECK(do_work_now(dev, cmd_pool, [&](vk::CommandBuffer& a_cmd_buf) {
		a_cmd_buf.buildAccelerationStructuresKHR(geom_build_info, &build_range_info);
		}));

	return VulkanAccelStructInfo{
		{std::move(accel)},
		std::move(accel_buf),
		std::move(scratch_buf),
		geom_build_info.flags,
		std::vector<vk::AccelerationStructureGeometryKHR>{
			geom_build_info.pGeometries, geom_build_info.pGeometries + geom_build_info.geometryCount
		}
	};
}

[[nodiscard]] auto PTS::VulkanBottomAccelStructInfo::create(
	VulkanDeviceInfo const& dev,
	VulkanCmdPoolInfo const& cmd_pool,
	RenderableObject const& obj
) -> tl::expected<VulkanBottomAccelStructInfo, std::string> {
	// note: transform will be later applied to the instances of the acceleration structure
	// so it's not needed here
	auto vert_buf = VulkanBufferInfo{};
	auto index_buf = VulkanBufferInfo{};

	auto vertices = std::vector<VulkanRayTracingShaders::VertexData>{};
	vertices.reserve(obj.get_vertices().size());
	std::transform(obj.get_vertices().begin(), obj.get_vertices().end(), std::back_inserter(vertices),
	               [](auto const& vert) {
		               return VulkanRayTracingShaders::VertexData{vert};
	               }
	);
	TL_TRY_ASSIGN(vert_buf, VulkanBufferInfo::create(
		              dev,
		              VulkanBufferInfo::Type::AccelInput,
		              vertices.size() * sizeof(VulkanRayTracingShaders::VertexData),
		              tcb::make_span(vertices)
	              ));
	TL_TRY_ASSIGN(index_buf, VulkanBufferInfo::create(
		              dev,
		              VulkanBufferInfo::Type::AccelInput,
		              obj.get_indices().size() * sizeof(decltype(obj.get_indices())::value_type),
		              tcb::make_span(obj.get_indices())
	              ));
	auto prim_cnt = static_cast<uint32_t>(obj.get_indices().size() / 3);
	auto triangle_data = vk::AccelerationStructureGeometryTrianglesDataKHR{}
	                     .setVertexFormat(vk::Format::eR32G32B32Sfloat)
	                     .setVertexData(vert_buf.get_device_addr())
	                     .setVertexStride(sizeof(VulkanRayTracingShaders::VertexData))
	                     .setMaxVertex(static_cast<uint32_t>(vertices.size()))
	                     .setIndexType(vk::IndexType::eUint32)
	                     .setIndexData(index_buf.get_device_addr());
	auto geometry = vk::AccelerationStructureGeometryKHR{}
	                .setGeometryType(vk::GeometryTypeKHR::eTriangles)
	                .setFlags(vk::GeometryFlagBitsKHR::eOpaque)
	                .setGeometry(vk::AccelerationStructureGeometryDataKHR{triangle_data});
	auto build_info = vk::AccelerationStructureBuildGeometryInfoKHR{}
	                  .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
		                  vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess)
	                  .setGeometries(geometry)
	                  .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
	                  .setType(vk::AccelerationStructureTypeKHR::eBottomLevel);

	auto accel = VulkanAccelStructInfo{};
	TL_TRY_ASSIGN(accel,
	              VulkanAccelStructInfo::create(dev, cmd_pool, build_info, prim_cnt, prim_cnt)
	);

	return VulkanBottomAccelStructInfo{
		std::move(accel),
		std::move(vert_buf),
		std::move(index_buf),
	};
}

[[nodiscard]] auto PTS::VulkanTopAccelStructInfo::create(
	VulkanDeviceInfo const& dev,
	VulkanCmdPoolInfo const& cmd_pool
) -> tl::expected<VulkanTopAccelStructInfo, std::string> {
	auto accel_ins_vec = std::vector<vk::AccelerationStructureInstanceKHR>{};
	auto bottom_accels = std::vector<VulkanBottomAccelStructInfo>{};
	auto accel_ins_buf = VulkanBufferInfo{};
	TL_TRY_ASSIGN(accel_ins_buf, VulkanBufferInfo::create(
		              dev,
		              VulkanBufferInfo::Type::AccelInput,
		              sizeof(decltype(accel_ins_vec)::value_type) * k_max_objs,
		              tcb::make_span(accel_ins_vec)
	              ));
	auto ins_cnt = static_cast<uint32_t>(accel_ins_vec.size());
	auto instance_data = vk::AccelerationStructureGeometryInstancesDataKHR{}
	                     .setArrayOfPointers(false)
	                     .setData(accel_ins_buf.get_device_addr());
	auto geometry = vk::AccelerationStructureGeometryKHR{}
	                .setGeometryType(vk::GeometryTypeKHR::eInstances)
	                .setFlags(vk::GeometryFlagBitsKHR::eOpaque)
	                .setGeometry(vk::AccelerationStructureGeometryDataKHR{instance_data});
	auto build_info = vk::AccelerationStructureBuildGeometryInfoKHR{}
	                  .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
		                  vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess |
		                  vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate)
	                  .setGeometries(geometry)
	                  .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
	                  .setType(vk::AccelerationStructureTypeKHR::eTopLevel);
	auto accel = VulkanAccelStructInfo{};
	TL_TRY_ASSIGN(accel, VulkanAccelStructInfo::create(
		              dev, cmd_pool,
		              build_info,
		              ins_cnt,
		              k_max_objs
	              ));
	return VulkanTopAccelStructInfo{
		std::move(accel),
		std::move(accel_ins_buf),
		std::move(bottom_accels),
		std::move(accel_ins_vec),
		dev, cmd_pool
	};
}

[[nodiscard]] auto PTS::VulkanTopAccelStructInfo::add_instance(
	VulkanBottomAccelStructInfo&& bottom_accel,
	glm::mat4 const& transform
) noexcept -> tl::expected<size_t, std::string> {
	if (m_instances.size() >= k_max_objs) {
		return TL_ERROR("max instances reached");
	}
	auto accel_ins = vk::AccelerationStructureInstanceKHR{}
	                 .setMask(0xFF)
	                 .setInstanceShaderBindingTableRecordOffset(0)
	                 .setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable)
	                 .setAccelerationStructureReference(bottom_accel.accel.storage_mem.get_device_addr());

	auto idx = size_t{0};
	if (m_free_idx.empty()) {
		idx = m_instances.size();

		accel_ins.setInstanceCustomIndex(idx);
		m_instances.emplace_back(accel_ins);
		m_bottom_accels.emplace_back(std::move(bottom_accel));
	} else {
		idx = m_free_idx.back();
		m_free_idx.pop_back();

		accel_ins.setInstanceCustomIndex(idx);
		m_instances[idx] = accel_ins;
		m_bottom_accels[idx] = std::move(bottom_accel);
	}

	TL_CHECK(update_instance_gpu(idx, transform));
	TL_CHECK(update_accel_gpu(
		// has to do a full rebuild because a new instance is added
		vk::BuildAccelerationStructureModeKHR::eBuild,
		0, m_instances.size() - 1
	));
	return idx;
}

[[nodiscard]] auto PTS::VulkanTopAccelStructInfo::remove_instance(size_t idx) noexcept
	-> tl::expected<void, std::string> {
	// clear the instance data
	m_instances[idx] = vk::AccelerationStructureInstanceKHR{};
	TL_CHECK_AND_PASS(update_instance_gpu(idx, glm::mat4{ 1.0f }));
	TL_CHECK_AND_PASS(update_accel_gpu(
		// has to do a full rebuild because the instance is removed
		vk::BuildAccelerationStructureModeKHR::eBuild,
		0, m_instances.size() - 1
	));

	m_free_idx.emplace_back(idx);
	return {};
}

[[nodiscard]] auto PTS::VulkanTopAccelStructInfo::update_instance_transform(
	size_t idx, glm::mat4 const& transform) noexcept
	-> tl::expected<void, std::string> {
	TL_CHECK_AND_PASS(update_instance_gpu(idx, transform));
	return update_accel_gpu(
		vk::BuildAccelerationStructureModeKHR::eUpdate,
		0, m_instances.size() - 1
	);
}

[[nodiscard]] auto PTS::VulkanTopAccelStructInfo::to_mat4x3(glm::mat4 const& mat) noexcept -> vk::TransformMatrixKHR {
	// vk::TransformMatrixKHR is a 3x4 matrix and is row-major
	// glm::mat4 is a 4x4 matrix and is column-major

	return vk::TransformMatrixKHR{
		std::array{
			std::array{mat[0][0], mat[1][0], mat[2][0], mat[3][0]},
			std::array{mat[0][1], mat[1][1], mat[2][1], mat[3][1]},
			std::array{mat[0][2], mat[1][2], mat[2][2], mat[3][2]}
		}
	};
}

[[nodiscard]] auto PTS::VulkanTopAccelStructInfo::update_instance_gpu(
	size_t idx,
	glm::mat4 const& transform
) noexcept -> tl::expected<void, std::string> {
	if (idx >= m_instances.size()) {
		return TL_ERROR("invalid instance index");
	}
	if (!m_dev || !*m_dev) {
		return TL_ERROR("device not set or invalid");
	}
	auto& dev = *m_dev;
	auto offset = static_cast<uint32_t>(idx * sizeof(vk::AccelerationStructureInstanceKHR));
	m_instances[idx].setTransform(to_mat4x3(transform));

	// update instance data
	return m_ins_mem.upload(m_instances[idx], offset);
}

[[nodiscard]] auto PTS::VulkanTopAccelStructInfo::update_accel_gpu(
	vk::BuildAccelerationStructureModeKHR build_type,
	size_t from, size_t to
) noexcept -> tl::expected<void, std::string> {
	if (from >= m_instances.size() || to >= m_instances.size() || from > to) {
		return TL_ERROR("invalid instance index");
	}
	if (!m_dev || !*m_dev) {
		return TL_ERROR("device not set or invalid");
	}
	if (!m_cmd_pool || !*m_cmd_pool) {
		return TL_ERROR("command pool not set or invalid");
	}

	auto& dev = *m_dev;
	auto& cmd_pool = *m_cmd_pool;

	// only build this instance
	auto build_range_info = vk::AccelerationStructureBuildRangeInfoKHR{}
	                        .setFirstVertex(0)
	                        .setPrimitiveCount(static_cast<uint32_t>(to - from + 1))
	                        .setPrimitiveOffset(
		                        static_cast<uint32_t>(from * sizeof(vk::AccelerationStructureInstanceKHR)))
	                        .setTransformOffset(0);
	if (m_accel.geometries[0].geometry.instances.data.deviceAddress != m_ins_mem.get_device_addr()) {
		return TL_ERROR("instance buffer address mismatch");
	}

	auto build_info = vk::AccelerationStructureBuildGeometryInfoKHR{}
	                  .setFlags(m_accel.flags)
	                  .setGeometries(m_accel.geometries)
	                  .setMode(build_type)
	                  .setType(vk::AccelerationStructureTypeKHR::eTopLevel)
	                  .setSrcAccelerationStructure(*m_accel)
	                  .setDstAccelerationStructure(*m_accel)
	                  .setScratchData(m_accel.scratch_mem.get_device_addr());

	TL_CHECK_AND_PASS(do_work_now(dev, cmd_pool, [&](vk::CommandBuffer& a_cmd_buf) {
		a_cmd_buf.buildAccelerationStructuresKHR(build_info, &build_range_info);
		}));

	return {};
}
