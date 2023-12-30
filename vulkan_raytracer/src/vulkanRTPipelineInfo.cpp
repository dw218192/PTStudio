#include "vulkanRTPipelineInfo.h"
#include "embeddedRes.h"

#include <vector>
#include <shaderc/shaderc.hpp>
#include <cmrc/cmrc.hpp>

CMRC_DECLARE(vulkan_raytracer_resources);
CMRC_DECLARE(core_resources);

// utility functions
namespace {
	// includer for compiling glsl shaders
	struct EmbeddedFsIncluder : shaderc::CompileOptions::IncluderInterface {
		~EmbeddedFsIncluder() override = default;
		EmbeddedFsIncluder() = default;
		DEFAULT_COPY_MOVE(EmbeddedFsIncluder);

		auto GetInclude(char const* requested_source, shaderc_include_type type,
		                char const* requesting_source, size_t include_depth) -> shaderc_include_result* override {
			auto const file_path = "shaders/" + std::string{requested_source};
			auto inc_src_res = tl::expected<std::string, std::string>{};
			if (type == shaderc_include_type_standard) {
				inc_src_res = PTS::try_get_embedded_res(cmrc::core_resources::get_filesystem(), file_path);
			} else {
				inc_src_res = PTS::try_get_embedded_res(cmrc::vulkan_raytracer_resources::get_filesystem(), file_path);
			}

			if (!inc_src_res) {
				m_res.source_name = "";
				m_res.source_name_length = 0;

				auto const content = copy_str(inc_src_res.error().c_str());
				m_res.content = content.data();
				m_res.content_length = content.size();
			} else {
				m_res.source_name = requested_source;
				m_res.source_name_length = strlen(requested_source);

				auto const content = copy_str(inc_src_res.value().c_str());
				m_res.content = content.data();
				m_res.content_length = content.size();
			}
			return &m_res;
		}

		auto ReleaseInclude(shaderc_include_result* data) -> void override { }

	private:
		auto copy_str(char const* src) -> std::string_view {
			auto const size = strlen(src) + 1;
			m_owned_mem.emplace_back(std::make_unique<char[]>(size));
			std::memcpy(m_owned_mem.back().get(), src, size);
			return std::string_view{m_owned_mem.back().get(), size - 1};
		}

		std::vector<std::unique_ptr<char[]>> m_owned_mem;
		shaderc_include_result m_res{};
	};

	[[nodiscard]] auto create_shader_glsl(
		PTS::VulkanDeviceInfo const& dev,
		std::string const& src,
		std::string_view name,
		vk::ShaderStageFlagBits stage
	) -> tl::expected<PTS::VulkanShaderInfo, std::string> {
		auto to_shaderc_stage = [](vk::ShaderStageFlagBits stage) {
			switch (stage) {
			case vk::ShaderStageFlagBits::eVertex:
				return shaderc_shader_kind::shaderc_vertex_shader;
			case vk::ShaderStageFlagBits::eFragment:
				return shaderc_shader_kind::shaderc_fragment_shader;
			case vk::ShaderStageFlagBits::eCompute:
				return shaderc_shader_kind::shaderc_compute_shader;
			case vk::ShaderStageFlagBits::eRaygenKHR:
				return shaderc_shader_kind::shaderc_raygen_shader;
			case vk::ShaderStageFlagBits::eAnyHitKHR:
				return shaderc_shader_kind::shaderc_anyhit_shader;
			case vk::ShaderStageFlagBits::eClosestHitKHR:
				return shaderc_shader_kind::shaderc_closesthit_shader;
			case vk::ShaderStageFlagBits::eMissKHR:
				return shaderc_shader_kind::shaderc_miss_shader;
			case vk::ShaderStageFlagBits::eIntersectionKHR:
				return shaderc_shader_kind::shaderc_intersection_shader;
			case vk::ShaderStageFlagBits::eCallableKHR:
				return shaderc_shader_kind::shaderc_callable_shader;
			default:
				return shaderc_shader_kind::shaderc_glsl_infer_from_source;
			}
		};

		try {
			auto const compiler = shaderc::Compiler{};
			auto options = shaderc::CompileOptions{};
			options.SetSourceLanguage(shaderc_source_language_glsl);
			options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
			options.SetTargetSpirv(shaderc_spirv_version_1_5);
			options.SetOptimizationLevel(shaderc_optimization_level_performance);
			options.SetIncluder(std::make_unique<EmbeddedFsIncluder>());
			auto const prep_res = compiler.PreprocessGlsl(
				src,
				to_shaderc_stage(stage),
				name.data(),
				options
			);
			if (prep_res.GetCompilationStatus() != shaderc_compilation_status_success) {
				return TL_ERROR("shader preprocessing failed:\n{}", prep_res.GetErrorMessage());
			}

			auto const prepped_src = std::string{prep_res.begin(), prep_res.end()};
			auto const res = compiler.CompileGlslToSpv(
				prepped_src,
				to_shaderc_stage(stage),
				name.data(),
				options
			);
			if (res.GetCompilationStatus() != shaderc_compilation_status_success) {
				return TL_ERROR("shader compilation failed:\n{}\n{}", res.GetErrorMessage(),
				                prepped_src);
			}
			auto const sprv_code = std::vector<uint32_t>{res.cbegin(), res.cend()};
			auto shader = dev->createShaderModuleUnique(
				vk::ShaderModuleCreateInfo{
					vk::ShaderModuleCreateFlags{},
					std::distance(sprv_code.cbegin(), sprv_code.cend()) * sizeof(uint32_t),
					sprv_code.data()
				}
			);
			return PTS::VulkanShaderInfo{{std::move(shader)}, stage};
		} catch (vk::SystemError& err) {
			return TL_ERROR("vulkan error:\n{} ", err.what());
		}
	}
}


auto PTS::VulkanRTPipelineInfo::create(
	VulkanDeviceInfo const& dev,
	VulkanCmdPoolInfo const& cmd_pool,
	VulkanImageInfo const& output_img,
	VulkanDescSetPoolInfo const& desc_set_pool
) -> tl::expected<VulkanRTPipelineInfo, std::string> {
	auto vk_top_accel = VulkanTopAccelStructInfo{};
	TL_TRY_ASSIGN(vk_top_accel, VulkanTopAccelStructInfo::create(dev, cmd_pool));

	auto ray_gen_shader = VulkanShaderInfo{};
	auto miss_shader = VulkanShaderInfo{};
	auto chit_shader = VulkanShaderInfo{};

	auto shader_infos = std::array<VulkanShaderInfo, 3>{};

	auto rgen_shader_src = std::string{};
	auto miss_shader_src = std::string{};
	auto chit_shader_src = std::string{};

	auto const fs = cmrc::vulkan_raytracer_resources::get_filesystem();
	TL_TRY_ASSIGN(rgen_shader_src, PTS::try_get_embedded_res(fs, k_rgen_shader_path));
	TL_TRY_ASSIGN(miss_shader_src, PTS::try_get_embedded_res(fs, k_miss_shader_path));
	TL_TRY_ASSIGN(chit_shader_src, PTS::try_get_embedded_res(fs, k_chit_shader_path));

	TL_TRY_ASSIGN(shader_infos[0], create_shader_glsl(
		              dev,
		              rgen_shader_src,
		              "rgen_shader",
		              vk::ShaderStageFlagBits::eRaygenKHR
	              ));
	TL_TRY_ASSIGN(shader_infos[1], create_shader_glsl(
		              dev,
		              miss_shader_src,
		              "miss_shader",
		              vk::ShaderStageFlagBits::eMissKHR
	              ));
	TL_TRY_ASSIGN(shader_infos[2], create_shader_glsl(
		              dev,
		              chit_shader_src,
		              "chit_shader",
		              vk::ShaderStageFlagBits::eClosestHitKHR
	              ));

	auto shader_stages = std::array{
		vk::PipelineShaderStageCreateInfo{
			vk::PipelineShaderStageCreateFlags{},
			vk::ShaderStageFlagBits::eRaygenKHR,
			*shader_infos[0],
			"main",
			nullptr
		},
		vk::PipelineShaderStageCreateInfo{
			vk::PipelineShaderStageCreateFlags{},
			vk::ShaderStageFlagBits::eMissKHR,
			*shader_infos[1],
			"main",
			nullptr
		},
		vk::PipelineShaderStageCreateInfo{
			vk::PipelineShaderStageCreateFlags{},
			vk::ShaderStageFlagBits::eClosestHitKHR,
			*shader_infos[2],
			"main",
			nullptr
		},
	};
	auto shader_groups = std::array{
		// group0  raygen 
		vk::RayTracingShaderGroupCreateInfoKHR{}
		.setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
		.setGeneralShader(0)
		.setClosestHitShader(VK_SHADER_UNUSED_KHR)
		.setAnyHitShader(VK_SHADER_UNUSED_KHR)
		.setIntersectionShader(VK_SHADER_UNUSED_KHR),
		// group1 miss
		vk::RayTracingShaderGroupCreateInfoKHR{}
		.setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
		.setGeneralShader(1)
		.setClosestHitShader(VK_SHADER_UNUSED_KHR)
		.setAnyHitShader(VK_SHADER_UNUSED_KHR)
		.setIntersectionShader(VK_SHADER_UNUSED_KHR),
		// group2 chit
		vk::RayTracingShaderGroupCreateInfoKHR{}
		.setType(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup)
		.setGeneralShader(VK_SHADER_UNUSED_KHR)
		.setClosestHitShader(2)
		.setAnyHitShader(VK_SHADER_UNUSED_KHR)
		.setIntersectionShader(VK_SHADER_UNUSED_KHR)
	};

	try {
		auto accel_info = vk_top_accel.get_accel().get_desc_info();
		auto img_info = output_img.get_desc_info();

		// create material buffer
		auto mat_buf = VulkanBufferInfo{};
		TL_TRY_ASSIGN(mat_buf, VulkanBufferInfo::create(
			              dev,
			              VulkanBufferInfo::Type::Uniform,
			              sizeof(VulkanRayTracingShaders::MaterialData) * k_max_instances
		              ));
		auto mat_buf_info = mat_buf.get_desc_info();

		// create lights buffer
		auto lights_buf = VulkanBufferInfo{};
		TL_TRY_ASSIGN(lights_buf, VulkanBufferInfo::create(
			              dev,
			              VulkanBufferInfo::Type::Uniform,
			              VulkanRayTracingShaders::LightBlock::get_offset(k_max_lights)
		              ));
		auto lights_buf_info = lights_buf.get_desc_info();

		// null buffer for unbounded arrays
		auto min_alignment = dev.physical_device.getProperties().limits.minStorageBufferOffsetAlignment;
		auto null_buf = VulkanBufferInfo{};
		TL_TRY_ASSIGN(null_buf, VulkanBufferInfo::create(
			              dev,
			              VulkanBufferInfo::Type::Storage,
			              min_alignment
		              ));
		auto null_buf_infos = std::array<vk::DescriptorBufferInfo, k_max_instances>{};
		std::fill(null_buf_infos.begin(), null_buf_infos.end(), null_buf.get_desc_info());

		// create descriptor sets
		auto desc_sets = std::array<VulkanDescSetInfo, VulkanRayTracingShaders::RayTracingBindings::k_num_sets>{};
		auto cur_set = 0;

		// set 0
		desc_sets[cur_set++]
			.add_binding(
				vk::DescriptorSetLayoutBinding{}
				.setBinding(VulkanRayTracingShaders::RayTracingBindings::ACCEL_STRUCT_BINDING.binding)
				.setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR),
				vk::DescriptorBindingFlags{},
				vk::WriteDescriptorSet{}
				.setPNext(&accel_info)
			)
			.add_binding(
				vk::DescriptorSetLayoutBinding{}
				.setBinding(VulkanRayTracingShaders::RayTracingBindings::OUTPUT_IMAGE_BINDING.binding)
				.setDescriptorType(vk::DescriptorType::eStorageImage)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR),
				vk::DescriptorBindingFlags{},
				vk::WriteDescriptorSet{}
				.setImageInfo(img_info)
			)
			.add_binding(
				vk::DescriptorSetLayoutBinding{}
				.setBinding(VulkanRayTracingShaders::RayTracingBindings::MATERIALS_BINDING.binding)
				.setDescriptorType(vk::DescriptorType::eUniformBuffer)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eClosestHitKHR),
				vk::DescriptorBindingFlags{},
				vk::WriteDescriptorSet{}
				.setBufferInfo(mat_buf_info)
			)
			.add_binding(
				vk::DescriptorSetLayoutBinding{}
				.setBinding(VulkanRayTracingShaders::RayTracingBindings::LIGHTS_BINDING.binding)
				.setDescriptorType(vk::DescriptorType::eUniformBuffer)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eClosestHitKHR),
				vk::DescriptorBindingFlags{},
				vk::WriteDescriptorSet{}
				.setBufferInfo(lights_buf_info)
			);

		TL_CHECK(desc_sets[cur_set - 1].create(dev, desc_set_pool));

		// desc set for unbounded arrays
		desc_sets[cur_set++]
			.add_binding(
				vk::DescriptorSetLayoutBinding{}
				.setBinding(VulkanRayTracingShaders::RayTracingBindings::VERTEX_ATTRIBS_BINDING.binding)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setDescriptorCount(k_max_instances)
				.setStageFlags(vk::ShaderStageFlagBits::eClosestHitKHR),
				vk::DescriptorBindingFlags{
					vk::DescriptorBindingFlagBits::ePartiallyBound |
					vk::DescriptorBindingFlagBits::eVariableDescriptorCount |
					vk::DescriptorBindingFlagBits::eUpdateAfterBind
				},
				vk::WriteDescriptorSet{}
				.setBufferInfo(null_buf_infos)
			);

		TL_CHECK(desc_sets[cur_set - 1].create(dev, desc_set_pool));

		desc_sets[cur_set++]
			.add_binding(
				vk::DescriptorSetLayoutBinding{}
				.setBinding(VulkanRayTracingShaders::RayTracingBindings::INDICES_BINDING.binding)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setDescriptorCount(k_max_instances)
				.setStageFlags(vk::ShaderStageFlagBits::eClosestHitKHR),
				vk::DescriptorBindingFlags{
					vk::DescriptorBindingFlagBits::ePartiallyBound |
					vk::DescriptorBindingFlagBits::eVariableDescriptorCount |
					vk::DescriptorBindingFlagBits::eUpdateAfterBind
				},
				vk::WriteDescriptorSet{}
				.setBufferInfo(null_buf_infos)
			);

		TL_CHECK(desc_sets[cur_set - 1].create(dev, desc_set_pool));

		if (cur_set != VulkanRayTracingShaders::RayTracingBindings::k_num_sets) {
			return TL_ERROR("not creating all descriptor sets, expected: {}, created: {}",
			                VulkanRayTracingShaders::RayTracingBindings::k_num_sets, cur_set);
		}

		// push constants
		auto push_const_range = vk::PushConstantRange{}
		                        .setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR)
		                        .setOffset(0)
		                        .setSize(sizeof(VulkanRayTracingShaders::PerFrameData));

		auto desc_set_layouts = std::array<vk::DescriptorSetLayout,
		                                   VulkanRayTracingShaders::RayTracingBindings::k_num_sets>{};
		std::transform(desc_sets.begin(), desc_sets.end(), desc_set_layouts.begin(),
		               [](VulkanDescSetInfo const& info) {
			               return info.get_layout();
		               }
		);

		auto pipeline_layout = dev->createPipelineLayoutUnique(
			vk::PipelineLayoutCreateInfo{}
			.setSetLayouts(desc_set_layouts)
			.setPushConstantRanges(push_const_range)
		);
		auto pipeline = dev->createRayTracingPipelineKHRUnique(
			nullptr, nullptr,
			vk::RayTracingPipelineCreateInfoKHR{}
			.setStages(shader_stages)
			.setGroups(shader_groups)
			.setMaxPipelineRayRecursionDepth(1) // for possible visibility tests in the chit shader
			.setLayout(*pipeline_layout)
		);
		if (pipeline.result != vk::Result::eSuccess) {
			return TL_ERROR("failed to create ray tracing pipeline");
		}

		// create descriptor set
		auto props = dev.physical_device.getProperties2<
			vk::PhysicalDeviceProperties2,
			vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
		auto rt_props = props.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

		auto handle_size = rt_props.shaderGroupHandleSize;
		auto handle_size_aligned = rt_props.shaderGroupHandleAlignment;
		auto group_count = static_cast<uint32_t>(shader_groups.size());
		auto sbt_size = handle_size_aligned * group_count;


		// create SBT
		auto handles = std::vector<uint8_t>(sbt_size);
		if (dev->getRayTracingShaderGroupHandlesKHR(
			*pipeline.value,
			0, group_count,
			sbt_size,
			handles.data()
		) != vk::Result::eSuccess) {
			return TL_ERROR("failed to get ray tracing shader group handles");
		}

		auto raygen_buf = VulkanBufferInfo{};
		TL_TRY_ASSIGN(raygen_buf, VulkanBufferInfo::create(
			              dev,
			              VulkanBufferInfo::Type::ShaderBindingTable,
			              handle_size,
			              tcb::span{ handles.data(), 1 }
		              ));
		auto miss_buf = VulkanBufferInfo{};
		TL_TRY_ASSIGN(miss_buf, VulkanBufferInfo::create(
			              dev,
			              VulkanBufferInfo::Type::ShaderBindingTable,
			              handle_size,
			              tcb::span{ handles.data() + handle_size_aligned, 1 }
		              ));
		auto hit_buf = VulkanBufferInfo{};
		TL_TRY_ASSIGN(hit_buf, VulkanBufferInfo::create(
			              dev,
			              VulkanBufferInfo::Type::ShaderBindingTable,
			              handle_size,
			              tcb::span{ handles.data() + handle_size_aligned * 2, 1 }
		              ));

		auto raygen_region = vk::StridedDeviceAddressRegionKHR{}
		                     .setDeviceAddress(raygen_buf.get_device_addr())
		                     .setSize(handle_size_aligned)
		                     .setStride(handle_size_aligned);
		auto miss_region = vk::StridedDeviceAddressRegionKHR{}
		                   .setDeviceAddress(miss_buf.get_device_addr())
		                   .setSize(handle_size_aligned)
		                   .setStride(handle_size_aligned);
		auto hit_region = vk::StridedDeviceAddressRegionKHR{}
		                  .setDeviceAddress(hit_buf.get_device_addr())
		                  .setSize(handle_size_aligned)
		                  .setStride(handle_size_aligned);

		return VulkanRTPipelineInfo{
			{std::move(pipeline.value)},
			std::move(pipeline_layout),
			std::move(vk_top_accel),
			std::move(desc_sets),
			std::move(mat_buf),
			std::move(lights_buf),
			std::move(raygen_buf),
			std::move(miss_buf),
			std::move(hit_buf),
			raygen_region,
			miss_region,
			hit_region,
			std::move(null_buf)
		};
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

auto PTS::VulkanRTPipelineInfo::bind_indices(
	VulkanDeviceInfo const& dev,
	int index,
	tcb::span<unsigned const> indices
) noexcept -> tl::expected<void, std::string> {
	if (index >= k_max_instances || index < 0) {
		return TL_ERROR("index out of bounds");
	}
	if (indices.size() % 3 != 0) {
		return TL_ERROR("indices must be a multiple of 3");
	}

	// create an SSBO for the indices
	auto index_buf = VulkanBufferInfo{};
	auto index_data = std::vector<VulkanRayTracingShaders::FaceIndexData>{};
	index_data.reserve(indices.size() / 3);
	for (auto i = 0u; i < indices.size(); i += 3) {
		index_data.emplace_back(indices[i], indices[i + 1], indices[i + 2]);
	}

	TL_TRY_ASSIGN(index_buf, VulkanBufferInfo::create(
		              dev,
		              VulkanBufferInfo::Type::Storage,
		              sizeof(VulkanRayTracingShaders::FaceIndexData) * index_data.size(),
		              tcb::make_span(index_data)
	              ));
	auto index_buf_info = index_buf.get_desc_info();

	auto write_set = vk::WriteDescriptorSet{}
	                 .setDstSet(*desc_sets[VulkanRayTracingShaders::RayTracingBindings::INDICES_BINDING.set])
	                 .setDstBinding(VulkanRayTracingShaders::RayTracingBindings::INDICES_BINDING.binding)
	                 .setDescriptorCount(1)
	                 .setDstArrayElement(index)
	                 .setDescriptorType(vk::DescriptorType::eStorageBuffer)
	                 .setBufferInfo(index_buf_info);
	try {
		dev->updateDescriptorSets(write_set, {});
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}

	index_ssbos[index] = std::move(index_buf);
	return {};
}

auto PTS::VulkanRTPipelineInfo::bind_vertex_attribs(
	VulkanDeviceInfo const& dev,
	int index,
	tcb::span<Vertex const> vertices
) noexcept -> tl::expected<void, std::string> {
	if (index >= k_max_instances || index < 0) {
		return TL_ERROR("index out of bounds");
	}

	auto vertex_buf = VulkanBufferInfo{};
	auto vertex_attribs = std::vector<VulkanRayTracingShaders::VertexAttribData>{};
	vertex_attribs.reserve(vertices.size());
	std::transform(vertices.begin(), vertices.end(), std::back_inserter(vertex_attribs),
	               [](Vertex const& v) {
		               return VulkanRayTracingShaders::VertexAttribData{v};
	               }
	);

	TL_TRY_ASSIGN(vertex_buf, VulkanBufferInfo::create(
		              dev,
		              VulkanBufferInfo::Type::Storage,
		              sizeof(VulkanRayTracingShaders::VertexAttribData) * vertex_attribs.size(),
		              tcb::make_span(vertex_attribs)
	              ));

	auto vertex_buf_info = vertex_buf.get_desc_info();
	auto write_set = vk::WriteDescriptorSet{}
	                 .setDstSet(*desc_sets[VulkanRayTracingShaders::RayTracingBindings::VERTEX_ATTRIBS_BINDING.set])
	                 .setDstBinding(VulkanRayTracingShaders::RayTracingBindings::VERTEX_ATTRIBS_BINDING.binding)
	                 .setDescriptorCount(1)
	                 .setDstArrayElement(index)
	                 .setDescriptorType(vk::DescriptorType::eStorageBuffer)
	                 .setBufferInfo(vertex_buf_info);
	try {
		dev->updateDescriptorSets(write_set, {});
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}

	vertex_attribs_ssbos[index] = std::move(vertex_buf);
	return {};
}
