#include "vulkanRayTracingRenderer.h"

#include <GLFW/glfw3.h>
#include <shaderc/shaderc.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <tuple>

#include <imgui.h>

#include "application.h"
#include "imgui/reflectedField.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

static constexpr auto k_desc_pool_sizes = std::array{
	vk::DescriptorPoolSize{
		vk::DescriptorType::eAccelerationStructureKHR,
		1
	},
	vk::DescriptorPoolSize{
		vk::DescriptorType::eStorageBuffer,
		3 + 2 * PTS::k_max_instances
	},
	vk::DescriptorPoolSize{
		vk::DescriptorType::eStorageImage,
		1
	},
	vk::DescriptorPoolSize{
		vk::DescriptorType::eUniformBuffer,
		1
	}
};

// conversion and convenience functions
auto to_cstr_vec(tcb::span<std::string_view> views) -> std::vector<char const*> {
	auto res = std::vector<char const*>{};
	res.reserve(views.size());
	std::transform(views.begin(), views.end(), std::back_inserter(res),
	               [](std::string_view view) {
		               return view.data();
	               }
	);
	return res;
}

auto has_ext(tcb::span<vk::ExtensionProperties const> props, std::string_view ext) -> bool {
	return std::find_if(props.begin(), props.end(),
	                    [&](vk::ExtensionProperties const& prop) {
		                    return ext == prop.extensionName;
	                    }
	) != props.end();
}

auto has_ext(tcb::span<vk::ExtensionProperties const> props, tcb::span<std::string_view> exts) -> bool {
	return std::all_of(exts.begin(), exts.end(),
	                   [&](std::string_view ext) {
		                   return has_ext(props, ext);
	                   }
	);
}

auto has_layer(tcb::span<vk::LayerProperties const> props, std::string_view layer) -> bool {
	return std::find_if(props.begin(), props.end(),
	                    [&](vk::LayerProperties const& prop) {
		                    return layer == prop.layerName;
	                    }
	) != props.end();
}

auto has_layer(tcb::span<vk::LayerProperties const> props, tcb::span<std::string_view> layers) -> bool {
	return std::all_of(layers.begin(), layers.end(),
	                   [&](std::string_view layer) {
		                   return has_layer(props, layer);
	                   }
	);
}

[[nodiscard]] auto create_instance(
	tcb::span<std::string_view> required_ins_ext,
	tcb::span<std::string_view> required_gl_ext
) -> tl::expected<PTS::VulkanInsInfo, std::string> {
	auto glfw_extensions_count = 0u;
	auto glfw_exts_raw = glfwGetRequiredInstanceExtensions(&glfw_extensions_count);
	auto ins_exts = std::vector<std::string_view>{
		glfw_exts_raw,
		glfw_exts_raw + glfw_extensions_count
	};
	for (auto const& ext : required_ins_ext) {
		if (std::find(ins_exts.begin(), ins_exts.end(), ext) == ins_exts.end()) {
			ins_exts.emplace_back(ext);
		}
	}

	// check if GL extensions are available
	{
		auto gl_exts = std::vector<std::string>{};
		auto no_gl_ext = 0u;
		glGetIntegerv(GL_NUM_EXTENSIONS, reinterpret_cast<GLint*>(&no_gl_ext));
		CHECK_GL_ERROR();

		for (auto i = 0u; i < no_gl_ext; ++i) {
			gl_exts.emplace_back(reinterpret_cast<char const*>(glGetStringi(GL_EXTENSIONS, i)));
			CHECK_GL_ERROR();
		}

		for (auto const& ext : required_gl_ext) {
			if (std::find(gl_exts.cbegin(), gl_exts.cend(), ext) == gl_exts.cend()) {
				return TL_ERROR("GL extension not found: " + std::string{ext});
			}
		}
	}

	try {
		// check if all required instance extensions are available
		auto const ext_props = vk::enumerateInstanceExtensionProperties();
		if (!has_ext(ext_props, required_ins_ext)) {
			return TL_ERROR("required instance extension not found");
		}

		// check if all layers are available
		auto layers = std::vector<std::string_view>{};
#ifndef NDEBUG
		layers.push_back("VK_LAYER_KHRONOS_validation");
#endif
		auto layer_props = vk::enumerateInstanceLayerProperties();
		if (!has_layer(layer_props, layers)) {
			return TL_ERROR("required layer not found");
		}

		constexpr auto app_info = vk::ApplicationInfo{
			"PTS::VulkanRayTracingRenderer",
			VK_MAKE_VERSION(1, 0, 0),
			"PTS",
			VK_MAKE_VERSION(1, 0, 0),
			VK_API_VERSION_1_2
		};

		auto ins_exts_cstr = to_cstr_vec(ins_exts);
		auto layers_cstr = to_cstr_vec(layers);
		auto ins = vk::createInstanceUnique(
			vk::InstanceCreateInfo{
				vk::InstanceCreateFlags{
					VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
				},
				&app_info,
				layers_cstr,
				ins_exts_cstr
			}
		);

		VULKAN_HPP_DEFAULT_DISPATCHER.init(*ins);
		return PTS::VulkanInsInfo{{std::move(ins)}, std::move(ins_exts), std::move(layers)};
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

template <typename CreateInfoChainType>
[[nodiscard]] auto create_device(
	PTS::VulkanInsInfo const& ins,
	tcb::span<std::string_view> required_device_ext,
	std::function<CreateInfoChainType(vk::DeviceCreateInfo)> create_info_chain
) -> tl::expected<PTS::VulkanDeviceInfo, std::string> {
	try {
		// get a physical device
		auto const physical_devices = ins->enumeratePhysicalDevices();
		if (physical_devices.empty()) {
			return TL_ERROR("no physical device found");
		}

		auto select_physical_device = [&](
			std::vector<vk::PhysicalDevice> const& devices) -> std::optional<vk::PhysicalDevice> {
			for (auto physical_device : devices) {
				// return the first device that supports all required extensions
				auto const ext_props = physical_device.enumerateDeviceExtensionProperties();
				if (has_ext(ext_props, required_device_ext)) {
					return physical_device;
				}
			}
			return std::nullopt;
		};

		auto physical_device = vk::PhysicalDevice{};
		if (auto const res = select_physical_device(physical_devices); !res) {
			return TL_ERROR("no suitable physical device found");
		} else {
			physical_device = *res;
		}

		// request a single graphics queue
		auto const queue_family_props = physical_device.getQueueFamilyProperties();
		auto queue_family_index = std::numeric_limits<uint32_t>::max();
		for (auto i = 0u; i < queue_family_props.size(); ++i) {
			if (queue_family_props[i].queueFlags & vk::QueueFlagBits::eGraphics) {
				queue_family_index = i;
				break;
			}
		}
		if (queue_family_index == std::numeric_limits<uint32_t>::max()) {
			return TL_ERROR("no queue family found");
		}
		auto const queue_priorities = std::array{0.0f};
		auto const queue_create_info = vk::DeviceQueueCreateInfo{
			vk::DeviceQueueCreateFlags{},
			queue_family_index,
			queue_priorities
		};
		// create logical device
		auto const dev_create_infos = std::array{queue_create_info};
		auto const layers_cstr = std::array<char const*, 0>{};
		auto const dev_exts_cstr = to_cstr_vec(required_device_ext);

		auto create_info = vk::DeviceCreateInfo{
			vk::DeviceCreateFlags{},
			dev_create_infos,
			layers_cstr,
			// ignored, see https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDeviceCreateInfo.html
			dev_exts_cstr
		};
		auto info_chain = create_info_chain(create_info);
		auto dev = physical_device.createDeviceUnique(info_chain.get<vk::DeviceCreateInfo>());
		VULKAN_HPP_DEFAULT_DISPATCHER.init(*dev);

		return PTS::VulkanDeviceInfo{{std::move(dev)}, physical_device, queue_family_index};
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

[[nodiscard]] auto create_cmd_pool(
	PTS::VulkanDeviceInfo const& dev) -> tl::expected<PTS::VulkanCmdPoolInfo, std::string> {
	try {
		auto cmd_pool = dev->createCommandPoolUnique(
			vk::CommandPoolCreateInfo{
				vk::CommandPoolCreateFlags{
					vk::CommandPoolCreateFlagBits::eResetCommandBuffer
				},
				dev.queue_family_idx,
			}
		);
		auto const queue = dev->getQueue(dev.queue_family_idx, 0);

		return PTS::VulkanCmdPoolInfo{{std::move(cmd_pool)}, queue};
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

[[nodiscard]] auto
create_desc_set_pool(PTS::VulkanDeviceInfo const& dev) -> tl::expected<PTS::VulkanDescSetPoolInfo, std::string> {
	try {
		auto pool = dev->createDescriptorPoolUnique(
			vk::DescriptorPoolCreateInfo{}
			.setFlags(
				vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
				vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind)
			.setMaxSets(PTS::VulkanRayTracingShaders::RayTracingBindings::k_num_sets)
			.setPoolSizes(k_desc_pool_sizes)
		);
		return PTS::VulkanDescSetPoolInfo{{std::move(pool)}};
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

[[nodiscard]] auto create_tex(
	PTS::VulkanDeviceInfo& dev,
	vk::Format fmt,
	unsigned width, unsigned height,
	vk::ImageUsageFlags usage_flags,
	vk::MemoryPropertyFlags prop_flags,
	vk::ImageAspectFlags aspect_flags,
	vk::ImageLayout initial_layout,
	vk::ImageLayout layout,
	vk::SamplerCreateInfo sampler_info,
	bool shared
) -> tl::expected<PTS::VulkanImageInfo, std::string> {
	try {
		auto img_info = vk::ImageCreateInfo{
			vk::ImageCreateFlags{},
			vk::ImageType::e2D,
			fmt,
			vk::Extent3D{width, height, 1},
			1,
			1,
			vk::SampleCountFlagBits::e1,
			vk::ImageTiling::eOptimal,
			usage_flags,
			vk::SharingMode::eExclusive,
			0,
			nullptr,
			initial_layout
		};
		auto shared_img = PTS::VulkanGLInteropUtils::SharedImage{};

		if (shared) {
			TL_TRY_ASSIGN(shared_img,
			              PTS::VulkanGLInteropUtils::create_shared_image(
				              *dev,
				              img_info,
				              prop_flags,
				              dev.physical_device.getMemoryProperties()
			              )
			);
		} else {
			shared_img.vk_image = dev->createImageUnique(img_info);
			shared_img.mem.vk_mem = dev->allocateMemoryUnique(
				vk::MemoryAllocateInfo{
					dev->getImageMemoryRequirements(*shared_img.vk_image).size,
					dev.physical_device.getMemoryProperties().memoryTypes[0].heapIndex
				}
			);
		}
		// bind image to memory
		dev->bindImageMemory(*shared_img.vk_image, *shared_img.mem.vk_mem, 0);
		auto view = dev->createImageViewUnique(
			vk::ImageViewCreateInfo{
				vk::ImageViewCreateFlags{},
				*shared_img.vk_image,
				vk::ImageViewType::e2D,
				fmt,
				vk::ComponentMapping{},
				vk::ImageSubresourceRange{
					aspect_flags,
					0, 1, 0, 1
				}
			}
		);

		auto sampler = dev->createSamplerUnique(sampler_info);
		return PTS::VulkanImageInfo{
			std::move(shared_img),
			std::move(view),
			layout,
			std::move(sampler)
		};
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}


[[nodiscard]] auto create_cmd_buf(
	PTS::VulkanDeviceInfo const& dev,
	PTS::VulkanCmdPoolInfo const& cmd_pool
) -> tl::expected<PTS::VulkanCmdBufInfo, std::string> {
	auto cmd_buf = vk::UniqueCommandBuffer{};
	try {
		auto cmd_bufs = dev->allocateCommandBuffersUnique(
			vk::CommandBufferAllocateInfo{
				cmd_pool.handle.get(),
				vk::CommandBufferLevel::ePrimary,
				1
			}
		);
		cmd_buf = std::move(cmd_bufs[0]);
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}

	return PTS::VulkanCmdBufInfo{{std::move(cmd_buf)}};
}

PTS::VulkanRayTracingRenderer::VulkanRayTracingRenderer(RenderConfig config)
	: Renderer{config, "Vulkan Ray Tracer"} {
	SceneObject::get_field_info<SceneObject::FieldTag::LOCAL_TRANSFORM>().get_on_change_callback_list()
		+= m_on_obj_local_trans_change;

	RenderableObject::get_field_info<RenderableObject::FieldTag::MAT>().get_on_change_callback_list()
		+= m_on_mat_change;

	m_light_data_link.get_on_push_back_callbacks() += [this](VulkanBufferInfo* buf, Light const*, LightData const&) {
		TL_CHECK_NON_FATAL(m_app, LogLevel::Error,
		                   buf->upload(tcb::span{ m_light_data_link.data(), m_light_data_link.size() }));
	};

	m_light_data_link.get_on_erase_callbacks() += [this](VulkanBufferInfo* buf, Light const*, LightData const&) {
		TL_CHECK_NON_FATAL(m_app, LogLevel::Error,
		                   buf->upload(tcb::span{ m_light_data_link.data(), m_light_data_link.size() }));
	};

	m_light_data_link.get_on_update_callbacks() += [this](VulkanBufferInfo* buf, Light const* light,
	                                                      LightData const& data) {
		auto id = size_t{};
		TL_TRY_ASSIGN_NON_FATAL(id, m_app, LogLevel::Error, m_light_data_link.get_idx(light));
		TL_CHECK_NON_FATAL(m_app, LogLevel::Info, buf->upload(data, id * sizeof(LightData)));
	};
}

PTS::VulkanRayTracingRenderer::~VulkanRayTracingRenderer() noexcept {
	SceneObject::get_field_info<SceneObject::FieldTag::LOCAL_TRANSFORM>().get_on_change_callback_list()
		-= m_on_obj_local_trans_change;

	RenderableObject::get_field_info<RenderableObject::FieldTag::MAT>().get_on_change_callback_list()
		-= m_on_mat_change;

	if (m_scene) {
		m_scene->get_callback_list(SceneChangeType::OBJECT_ADDED) -= m_on_add_obj;
		m_scene->get_callback_list(SceneChangeType::OBJECT_REMOVED) -= m_on_remove_obj;
	}
}

auto PTS::VulkanRayTracingRenderer::open_scene(Ref<Scene> scene) noexcept
	-> tl::expected<void, std::string> {
	if (!m_vk_device) {
		return TL_ERROR("renderer not initialized");
	}
	if (!m_vk_cmd_pool) {
		return TL_ERROR("command pool not initialized");
	}
	if (!m_vk_desc_set_pool) {
		return TL_ERROR("descriptor set pool not initialized");
	}
	if (!m_output_img.img.vk_image) {
		return TL_ERROR("output image not initialized");
	}

	if (m_scene) {
		m_scene->get_callback_list(SceneChangeType::OBJECT_ADDED) -= m_on_add_obj;
		m_scene->get_callback_list(SceneChangeType::OBJECT_REMOVED) -= m_on_remove_obj;
	}
	m_scene = &scene.get();
	m_scene->get_callback_list(SceneChangeType::OBJECT_ADDED) += m_on_add_obj;
	m_scene->get_callback_list(SceneChangeType::OBJECT_REMOVED) += m_on_remove_obj;

	if (*m_vk_pipeline) {
		// remove all objects
		for (auto const& kvp : m_rend_obj_data) {
			TL_CHECK_AND_PASS(remove_object(*kvp.first));
		}
	} else {
		// continue first time initialization, following init() call
		// create ray tracing pipeline
		TL_TRY_ASSIGN(m_vk_pipeline, VulkanRTPipelineInfo::create(
			              m_vk_device,
			              m_vk_cmd_pool,
			              m_output_img,
			              m_vk_desc_set_pool
		              ));
		// create command buffer
		TL_TRY_ASSIGN(m_vk_render_cmd_buf, create_cmd_buf(m_vk_device, m_vk_cmd_pool));
	}

	// set link to the light data
	m_light_data_link.get_user_data() = &m_vk_pipeline.lights_mem;

	// add objects
	for (auto& obj : scene.get().get_objects_of_type<RenderableObject>()) {
		TL_CHECK_AND_PASS(add_object(obj));
	}
	for (auto& obj : scene.get().get_objects_of_type<Light>()) {
		TL_CHECK_AND_PASS(add_object(obj));
	}
	// clear output image
	TL_CHECK_AND_PASS(reset_path_tracing());

	return {};
}

auto PTS::VulkanRayTracingRenderer::on_add_obj(Ref<SceneObject> obj) noexcept -> void {
	TL_CHECK_NON_FATAL(m_app, LogLevel::Error, add_object(obj));
	TL_CHECK_NON_FATAL(m_app, LogLevel::Error, reset_path_tracing());
}


auto PTS::VulkanRayTracingRenderer::on_remove_obj(Ref<SceneObject> obj) noexcept -> void {
	TL_CHECK_NON_FATAL(m_app, LogLevel::Error, remove_object(obj));
	TL_CHECK_NON_FATAL(m_app, LogLevel::Error, reset_path_tracing());
}

auto PTS::VulkanRayTracingRenderer::on_obj_local_trans_change(
	SceneObject::callback_data_t<SceneObject::FieldTag::LOCAL_TRANSFORM> data) -> void {
	if (auto const render_obj = data.obj.as<RenderableObject>()) {
		auto const it = m_rend_obj_data.find(render_obj);
		if (it != m_rend_obj_data.end()) {
			TL_CHECK_NON_FATAL(m_app, LogLevel::Error,
			                   m_vk_pipeline.top_accel.update_instance_transform(
				                   it->second.gpu_idx,
				                   render_obj->get_transform(TransformSpace::WORLD).get_matrix()
			                   )
			);
		}
	}
	TL_CHECK_NON_FATAL(m_app, LogLevel::Error, reset_path_tracing());
}

auto PTS::VulkanRayTracingRenderer::on_mat_change(
	RenderableObject::callback_data_t<RenderableObject::FieldTag::MAT> data) -> void {
	auto const it = m_rend_obj_data.find(&data.obj);
	if (it != m_rend_obj_data.end()) {
		auto mat_data = VulkanRayTracingShaders::MaterialData{data.obj.get_material()};
		TL_CHECK_NON_FATAL(m_app, LogLevel::Error,
		                   m_vk_pipeline.materials_mem.upload(
			                   mat_data,
			                   it->second.gpu_idx * sizeof(VulkanRayTracingShaders::MaterialData)
		                   )
		);
	} else {
		m_app->log(LogLevel::Error, "cannot find object {}", data.obj.get_name());
	}
	TL_CHECK_NON_FATAL(m_app, LogLevel::Error, reset_path_tracing());
}

auto PTS::VulkanRayTracingRenderer::render(View<Camera> camera) noexcept
	-> tl::expected<TextureHandle, std::string> {
	auto camera_data = VulkanRayTracingShaders::CameraData{camera.get()};
	if (m_path_tracing_data.camera_data && camera_data != m_path_tracing_data.camera_data) {
		if (*m_path_tracing_data.camera_data != camera_data) {
			TL_CHECK(reset_path_tracing());
		}
	}
	m_path_tracing_data.camera_data = camera_data;

	if (m_editing_data.unlimited_samples ||
		m_path_tracing_data.iteration < m_editing_data.num_samples) {
		// update per frame data
		auto const per_frame_data = VulkanRayTracingShaders::PerFrameData{
			camera_data,
			m_path_tracing_data.iteration,
			m_editing_data.num_samples,
			m_editing_data.max_bounces
		};
		try {
			m_vk_render_cmd_buf->reset(vk::CommandBufferResetFlags{});
			m_vk_render_cmd_buf->begin(vk::CommandBufferBeginInfo{});
			{
				// clear image
				m_vk_render_cmd_buf->bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *m_vk_pipeline);

				m_vk_render_cmd_buf->pushConstants(
					*m_vk_pipeline.layout,
					vk::ShaderStageFlagBits::eRaygenKHR,
					0,
					sizeof(VulkanRayTracingShaders::PerFrameData),
					&per_frame_data
				);

				m_vk_render_cmd_buf->bindDescriptorSets(
					vk::PipelineBindPoint::eRayTracingKHR,
					*m_vk_pipeline.layout,
					0,
					m_vk_pipeline.get_desc_sets(),
					nullptr
				);
				m_vk_render_cmd_buf->traceRaysKHR(
					m_vk_pipeline.raygen_region,
					m_vk_pipeline.miss_region,
					m_vk_pipeline.hit_region,
					{},
					m_config.width,
					m_config.height,
					1
				);
			}
			m_vk_render_cmd_buf->end();

			m_vk_cmd_pool.queue.submit(
				vk::SubmitInfo{}
				.setCommandBuffers(*m_vk_render_cmd_buf),
				nullptr
			);
			m_vk_cmd_pool.queue.waitIdle();

			++m_path_tracing_data.iteration;
		} catch (vk::SystemError& err) {
			return TL_ERROR(err.what());
		}
	}
	return m_output_img.img.gl_tex.get();
}

auto PTS::VulkanRayTracingRenderer::valid() const noexcept -> bool {
	return m_vk_ins && m_vk_device && m_vk_cmd_pool && m_vk_desc_set_pool && m_output_img.img.vk_image
		&& m_output_img.img.gl_tex.get() && m_vk_pipeline && m_vk_render_cmd_buf;
}

auto PTS::VulkanRayTracingRenderer::on_change_render_config() noexcept -> tl::expected<void, std::string> {
	return reset_path_tracing();
}

auto PTS::VulkanRayTracingRenderer::init(ObserverPtr<Application> app) noexcept
	-> tl::expected<void, std::string> {
	TL_CHECK(Renderer::init(app));

	auto ins_ext = VulkanGLInteropUtils::get_vk_ins_exts();
	ins_ext.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
	ins_ext.emplace_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

	auto gl_ext = VulkanGLInteropUtils::get_gl_exts();
	VULKAN_HPP_DEFAULT_DISPATCHER.init();
	TL_TRY_ASSIGN(m_vk_ins, create_instance(
		              tcb::make_span(ins_ext),
		              tcb::make_span(gl_ext)
	              ));

	auto device_ext = VulkanGLInteropUtils::get_vk_dev_exts();
	// add ray tracing extensions
	device_ext.emplace_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
	device_ext.emplace_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
	device_ext.emplace_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
	device_ext.emplace_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
	device_ext.emplace_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);

	// add descriptor indexing extension
	device_ext.emplace_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
	device_ext.emplace_back(VK_KHR_MAINTENANCE1_EXTENSION_NAME);

	using CreateInfoChainType = vk::StructureChain<
		vk::DeviceCreateInfo,
		vk::PhysicalDeviceBufferDeviceAddressFeatures,
		vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
		vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
		vk::PhysicalDeviceDescriptorIndexingFeaturesEXT
	>;
	TL_TRY_ASSIGN(m_vk_device, create_device(m_vk_ins, device_ext, std::function {
		              [](vk::DeviceCreateInfo create_info) {
		              return CreateInfoChainType {
		              create_info,
		              vk::PhysicalDeviceBufferDeviceAddressFeatures{}
		              .setBufferDeviceAddress(true),
		              vk::PhysicalDeviceRayTracingPipelineFeaturesKHR{}
		              .setRayTracingPipeline(true),
		              vk::PhysicalDeviceAccelerationStructureFeaturesKHR{}
		              .setAccelerationStructure(true),
		              vk::PhysicalDeviceDescriptorIndexingFeaturesEXT{}
		              .setDescriptorBindingPartiallyBound(true)
		              .setDescriptorBindingVariableDescriptorCount(true)
		              .setRuntimeDescriptorArray(true)
		              .setShaderSampledImageArrayNonUniformIndexing(true)
		              .setDescriptorBindingStorageBufferUpdateAfterBind(true)
		              .setShaderStorageBufferArrayNonUniformIndexing(true)
		              };
		              }
		              }));

	TL_TRY_ASSIGN(m_vk_cmd_pool, create_cmd_pool(m_vk_device));
	TL_TRY_ASSIGN(m_vk_desc_set_pool, create_desc_set_pool(m_vk_device));
	TL_TRY_ASSIGN(m_output_img, create_tex(
		              m_vk_device,
		              vk::Format::eR8G8B8A8Unorm,
		              k_max_width,
		              k_max_height,
		              vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::
		              eTransferDst,
		              vk::MemoryPropertyFlagBits::eDeviceLocal,
		              vk::ImageAspectFlagBits::eColor,
		              vk::ImageLayout::eUndefined,
		              vk::ImageLayout::eGeneral,
		              {},
		              true
	              ));

	// set up texture layout transition
	TL_CHECK_AND_PASS(do_work_now(m_vk_device, m_vk_cmd_pool, [this](vk::CommandBuffer& cmd_buf) {
		cmd_buf.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe,
			vk::PipelineStageFlagBits::eTransfer,
			vk::DependencyFlags{},
			nullptr,
			nullptr,
			vk::ImageMemoryBarrier{
			vk::AccessFlags{},
			vk::AccessFlagBits::eTransferWrite,
			vk::ImageLayout::eUndefined,
			m_output_img.layout,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			*m_output_img.img.vk_image,
			vk::ImageSubresourceRange{}
			.setAspectMask(vk::ImageAspectFlagBits::eColor)
			.setBaseMipLevel(0)
			.setLevelCount(1)
			.setBaseArrayLayer(0)
			.setLayerCount(1)
			}
		);
		}));
	return {};
}

auto PTS::VulkanRayTracingRenderer::draw_imgui() noexcept -> tl::expected<void, std::string> {
	ImGui::SetNextWindowSize({300, 100}, ImGuiCond_FirstUseEver);
	ImGui::Begin("Vulkan Ray Tracer");
	if (ImGui::ReflectedField("Editing Data", m_editing_data)) {
		TL_CHECK_AND_PASS(reset_path_tracing());
	}
	if (m_path_tracing_data.iteration < m_editing_data.num_samples || m_editing_data.unlimited_samples) {
		ImGui::Text("Iteration: %d", m_path_tracing_data.iteration);
	} else {
		ImGui::PushStyleColor(ImGuiCol_Text, {0.0f, 1.0f, 0.0f, 1.0f});
		ImGui::Text("Iteration: %d (done)", m_path_tracing_data.iteration);
		ImGui::PopStyleColor();
	}

	ImGui::End();
	return {};
}

auto PTS::VulkanRayTracingRenderer::reset_path_tracing() noexcept -> tl::expected<void, std::string> {
	m_path_tracing_data.iteration = 0;

	return {};
}

auto PTS::VulkanRayTracingRenderer::add_object(SceneObject const& obj) -> tl::expected<void, std::string> {
	if (auto render_obj = obj.as<RenderableObject>()) {
		if (m_rend_obj_data.count(render_obj)) {
			return TL_ERROR("object already added");
		}
		// create bottom level acceleration structure
		auto vk_bottom_accel = VulkanBottomAccelStructInfo{};
		TL_TRY_ASSIGN(vk_bottom_accel, VulkanBottomAccelStructInfo::create(m_vk_device, m_vk_cmd_pool, *render_obj));

		// add instance to top level acceleration structure
		auto id = size_t{0};
		TL_TRY_ASSIGN(id,
		              m_vk_pipeline.top_accel.add_instance(
			              std::move(vk_bottom_accel),
			              obj.get_transform(TransformSpace::WORLD).get_matrix()
		              )
		);

		// bind the vertex attributes and indices to the corresponding buffers
		auto const& vertex_attribs = render_obj->get_vertices();
		auto const& indices = render_obj->get_indices();

		TL_CHECK_AND_PASS(m_vk_pipeline.bind_vertex_attribs(m_vk_device, id, vertex_attribs));
		TL_CHECK_AND_PASS(m_vk_pipeline.bind_indices(m_vk_device, id, indices));

		// update material buffer
		auto mat_data = VulkanRayTracingShaders::MaterialData{render_obj->get_material()};
		TL_CHECK_AND_PASS(
			m_vk_pipeline.materials_mem.upload(
				mat_data,
				id * sizeof(VulkanRayTracingShaders::MaterialData)
			)
		);
		m_rend_obj_data.emplace(render_obj, PerObjectData{id});
	} else if (auto const light = obj.as<Light>()) {
		TL_CHECK(m_light_data_link.push_back(light, light->get_data()));
	} else {
		return TL_ERROR("unknown object of type: {}", obj.get_class_name());
	}

	return {};
}

auto PTS::VulkanRayTracingRenderer::remove_object(SceneObject const& obj) -> tl::expected<void, std::string> {
	if (auto const render_obj = obj.as<RenderableObject>()) {
		auto const it = m_rend_obj_data.find(render_obj);
		if (it == m_rend_obj_data.end()) {
			return TL_ERROR("object not found");
		}
		TL_CHECK_AND_PASS(m_vk_pipeline.top_accel.remove_instance(it->second.gpu_idx));
		m_rend_obj_data.erase(it);
	} else if (auto const light = obj.as<Light>()) {
		TL_CHECK_AND_PASS(m_light_data_link.erase(light));
	} else {
		return TL_ERROR("unknown object of type: {}", obj.get_class_name());
	}

	// no need to remove vertex attributes and indices because they will be overwritten
	return {};
}
