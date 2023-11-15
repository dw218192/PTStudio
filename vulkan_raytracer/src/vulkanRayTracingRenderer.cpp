#include "vulkanRayTracingRenderer.h"
#include <GLFW/glfw3.h>
#include <shaderc/shaderc.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <tuple>

using namespace PTS;

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

static constexpr float k_quad_data_pos_uv[] = {
	-1.0f, -1.0f, 0.0f,
    0.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
	1.0f, 0.0f,
	1.0f,  1.0f, 0.0f,
	1.0f, 1.0f,
	-1.0f, -1.0f, 0.0f,
    0.0f, 0.0f,
	1.0f,  1.0f, 0.0f,
	1.0f, 1.0f,
    -1.0f,  1.0f, 0.0f,
	0.0f, 1.0f
};

static constexpr int k_max_width = 4000;
static constexpr int k_max_height = 4000;

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
) -> tl::expected<VulkanInsInfo, std::string> {
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
        for (auto i = 0u; i < no_gl_ext; ++i) {
            gl_exts.emplace_back(reinterpret_cast<char const*>(glGetStringi(GL_EXTENSIONS, i)));
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
		return VulkanInsInfo{{std::move(ins)}, std::move(ins_exts), std::move(layers) };
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

template<typename CreateInfoChainType>
[[nodiscard]] auto create_device(
    VulkanInsInfo const& ins,
    tcb::span<std::string_view> required_device_ext,
    std::function<CreateInfoChainType(vk::DeviceCreateInfo)> create_info_chain
) -> tl::expected<VulkanDeviceInfo, std::string> {
	try {
		// get a physical device
		auto const physical_devices = ins->enumeratePhysicalDevices();
		if (physical_devices.empty()) {
			return TL_ERROR("no physical device found");
		}

        auto select_physical_device = [&](std::vector<vk::PhysicalDevice> const& devices) -> std::optional<vk::PhysicalDevice> {
            for (auto physical_device : devices) {
                // return the first device that supports all required extensions
                auto const ext_props = physical_device.enumerateDeviceExtensionProperties();
                if(has_ext(ext_props, required_device_ext)) {
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
		auto const queue_priorities = std::array{ 0.0f };
		auto const queue_create_info = vk::DeviceQueueCreateInfo{
			vk::DeviceQueueCreateFlags{},
			queue_family_index,
			queue_priorities
		};
		// create logical device
        auto const dev_create_infos = std::array { queue_create_info };
        auto const layers_cstr = std::array<char const*, 0> {};
        auto const dev_exts_cstr = to_cstr_vec(required_device_ext);

        auto create_info = vk::DeviceCreateInfo{
            vk::DeviceCreateFlags{},
            dev_create_infos,
            layers_cstr, // ignored, see https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDeviceCreateInfo.html
            dev_exts_cstr
        };
        auto info_chain = create_info_chain(create_info);
		auto dev = physical_device.createDeviceUnique(info_chain.get<vk::DeviceCreateInfo>());
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*dev);

		return VulkanDeviceInfo{{std::move(dev)}, physical_device, queue_family_index };
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

[[nodiscard]] auto create_cmd_pool(VulkanDeviceInfo const& dev) -> tl::expected<VulkanCmdPoolInfo, std::string> {
	try {
		auto cmd_pool = dev->createCommandPoolUnique(
			vk::CommandPoolCreateInfo{
				vk::CommandPoolCreateFlags { 
                    vk::CommandPoolCreateFlagBits::eResetCommandBuffer
                },
				dev.queue_family_idx,
			}
		);
		auto const queue = dev->getQueue(dev.queue_family_idx, 0);

		return VulkanCmdPoolInfo{{std::move(cmd_pool)}, queue };
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

[[nodiscard]] auto create_desc_set_pool(VulkanDeviceInfo const& dev) -> tl::expected<VulkanDescSetPoolInfo, std::string> {
    try {
        auto const pool_sizes = std::array{
            vk::DescriptorPoolSize{
                vk::DescriptorType::eAccelerationStructureKHR,
                1
            },
            vk::DescriptorPoolSize{
                vk::DescriptorType::eStorageBuffer,
                3
            },
            vk::DescriptorPoolSize{
                vk::DescriptorType::eStorageImage,
                1
            },
        };
        auto pool = dev->createDescriptorPoolUnique(
            vk::DescriptorPoolCreateInfo{}
                .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
                .setMaxSets(1)
                .setPoolSizes(pool_sizes)
        );
        return VulkanDescSetPoolInfo{{std::move(pool)}};
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

/**
 * @brief Convenience function to create a vulkan buffer
 * @param type the type of the buffer
 * @param size the size of the buffer, in bytes
 * @param data the data to be copied to the buffer, can be nullptr if the buffer is not to be initialized
*/
[[nodiscard]] auto create_buffer(
    VulkanDeviceInfo const& dev,
    VulkanBufferInfo::Type type,
    vk::DeviceSize size,
    void const* data
) -> tl::expected<VulkanBufferInfo, std::string> {
    try {
        auto usage_flags = vk::BufferUsageFlags{};
        auto mem_props_flags = vk::MemoryPropertyFlags{};
        switch (type) {
        case VulkanBufferInfo::Type::Scratch:
            usage_flags = vk::BufferUsageFlagBits::eStorageBuffer;
            mem_props_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
            break;
        case VulkanBufferInfo::Type::AccelInput:
            usage_flags = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                vk::BufferUsageFlagBits::eStorageBuffer;
            mem_props_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
            break;
        case VulkanBufferInfo::Type::AccelStorage:
            usage_flags = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR;
            mem_props_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
            break;
        case VulkanBufferInfo::Type::ShaderBindingTable:
            usage_flags = vk::BufferUsageFlagBits::eShaderBindingTableKHR;
            mem_props_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
            break;
        default:
            return TL_ERROR("invalid buffer type");
        }
        usage_flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;

        if(!size) {
            // create an empty buffer
            size = 1;
        }

        auto buffer = dev->createBufferUnique(
            vk::BufferCreateInfo{}
                .setSize(size)
                .setUsage(usage_flags)
                .setSharingMode(vk::SharingMode::eExclusive)
        );
        // get property index
        auto const mem_req = dev->getBufferMemoryRequirements(*buffer);
        auto mem_type_idx = std::numeric_limits<uint32_t>::max();
        auto mem_props = dev.physical_device.getMemoryProperties();
        for (auto i = 0u; i < mem_props.memoryTypeCount; ++i) {
            if ((mem_req.memoryTypeBits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & mem_props_flags) == mem_props_flags) {
                mem_type_idx = i;
                break;
            }
        }
        auto flags_info = vk::MemoryAllocateFlagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };
        auto alloc_info = vk::MemoryAllocateInfo{
            mem_req.size,
            mem_type_idx        
        };
        alloc_info.setPNext(&flags_info);
        auto mem = dev->allocateMemoryUnique(alloc_info);

        if (data) {
            auto const mapped = dev->mapMemory(*mem, 0, size);
            memcpy(mapped, data, size);
            dev->unmapMemory(*mem);
        }
        dev->bindBufferMemory(*buffer, *mem, 0);
        auto desc_info = vk::DescriptorBufferInfo{
            *buffer,
            0,
            size
        }; 
        auto device_addr = dev->getBufferAddressKHR(
            vk::BufferDeviceAddressInfo{
                *buffer
            }
        );
        return VulkanBufferInfo{{std::move(buffer)}, std::move(mem), std::move(desc_info), device_addr };
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

[[nodiscard]] auto do_work_now(
    VulkanDeviceInfo const& dev,
    VulkanCmdPoolInfo const& cmd_pool,
    std::function<void(vk::CommandBuffer&)> work
) -> tl::expected<void, std::string> {
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

[[nodiscard]] auto create_tex(
    VulkanDeviceInfo& dev,
    vk::Format fmt,
    unsigned width, unsigned height,
    vk::ImageUsageFlags usage_flags,
    vk::MemoryPropertyFlags prop_flags,
    vk::ImageAspectFlags aspect_flags,
    vk::ImageLayout initial_layout,
    vk::ImageLayout layout,
    vk::SamplerCreateInfo sampler_info,
    bool shared
) -> tl::expected<VulkanImageInfo, std::string> {
    try {
        auto img_info = vk::ImageCreateInfo{
            vk::ImageCreateFlags{},
            vk::ImageType::e2D,
            fmt,
            vk::Extent3D{ width, height, 1 },
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
        auto shared_img = VulkanGLInteropUtils::SharedImage{};

        if (shared) {
            TL_TRY_ASSIGN(shared_img,
                VulkanGLInteropUtils::create_shared_image(
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
        auto desc_img_info = vk::DescriptorImageInfo{
            *sampler,
            *view,
            layout
        };

        return VulkanImageInfo{
            std::move(shared_img),
            std::move(view),
            std::move(sampler),
            desc_img_info
        };

    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}


[[nodiscard]] auto create_shader_glsl(
    VulkanDeviceInfo const& dev,
    std::string_view src,
    std::string_view name,
    vk::ShaderStageFlagBits stage
) -> tl::expected<VulkanShaderInfo, std::string> {
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
        auto compiler = shaderc::Compiler{};
        auto options = shaderc::CompileOptions{};
        options.SetSourceLanguage(shaderc_source_language_glsl);
        options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
        options.SetTargetSpirv(shaderc_spirv_version_1_5);
        options.SetOptimizationLevel(shaderc_optimization_level_performance);
        auto const res = compiler.CompileGlslToSpv(
            src.data(),
            to_shaderc_stage(stage),
            name.data(),
            options
        );
        if (res.GetCompilationStatus() != shaderc_compilation_status_success) {
            return TL_ERROR(res.GetErrorMessage());
        }
        auto sprv_code = std::vector<uint32_t>{ res.cbegin(), res.cend() };
        auto shader = dev->createShaderModuleUnique(
            vk::ShaderModuleCreateInfo{
                vk::ShaderModuleCreateFlags{},
                std::distance(sprv_code.cbegin(), sprv_code.cend()) * sizeof(uint32_t),
                sprv_code.data()
            }
        );
        return VulkanShaderInfo{{std::move(shader)}, stage };
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

[[nodiscard]] auto config_cmd_buf(
    vk::CommandBuffer& cmd_buf,
    VulkanPipelineInfo const& pipeline,
    VulkanImageInfo const& output_img,
    unsigned width, unsigned height
) -> tl::expected<void, std::string> {
    try {
        cmd_buf.reset(vk::CommandBufferResetFlags{});
        cmd_buf.begin(vk::CommandBufferBeginInfo{});
        {
            cmd_buf.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
            cmd_buf.bindDescriptorSets(
                vk::PipelineBindPoint::eRayTracingKHR,
                *pipeline.layout,
                0,
                *pipeline.desc_set,
                nullptr
            );

            // set image layout
            auto const img_barrier = vk::ImageMemoryBarrier{}
                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setImage(*output_img.img.vk_image)
                .setSubresourceRange(
                    vk::ImageSubresourceRange{}
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseArrayLayer(0)
                        .setBaseMipLevel(0)
                        .setLayerCount(1)
                        .setLevelCount(1)
                )
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eGeneral);
            cmd_buf.pipelineBarrier(
                vk::PipelineStageFlagBits::eAllCommands,
                vk::PipelineStageFlagBits::eAllCommands,
                {}, {}, {}, img_barrier
            );

            cmd_buf.traceRaysKHR(
                pipeline.raygen_region,
                pipeline.miss_region,
                pipeline.hit_region,
                {},
                width,
                height,
                1
            );
        }
        cmd_buf.end();
        return {};
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

[[nodiscard]] auto create_cmd_buf(
    VulkanDeviceInfo const& dev,
    VulkanCmdPoolInfo const& cmd_pool,
    VulkanImageInfo const& output_img,
    VulkanPipelineInfo const& pipeline
) -> tl::expected<VulkanCmdBufInfo, std::string> {
    auto cmd_buf = vk::UniqueCommandBuffer {};
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

    return VulkanCmdBufInfo{{std::move(cmd_buf)}};
}

[[nodiscard]] auto create_accel(
    VulkanDeviceInfo const& dev,
    VulkanCmdPoolInfo const& cmd_pool,
    vk::AccelerationStructureBuildGeometryInfoKHR geom_build_info,
    vk::AccelerationStructureTypeKHR type,
    uint32_t primitive_count
) -> tl::expected<VulkanAccelStructInfo, std::string> {

    auto build_sizes = dev->getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice,
        geom_build_info,
        primitive_count
    );
    auto accel_buf = VulkanBufferInfo{};
    TL_TRY_ASSIGN(accel_buf, create_buffer(
        dev,
        VulkanBufferInfo::Type::AccelStorage,
        build_sizes.accelerationStructureSize,
        nullptr
    ));
    auto accel = dev->createAccelerationStructureKHRUnique(
        vk::AccelerationStructureCreateInfoKHR{}
            .setBuffer(*accel_buf)
            .setSize(build_sizes.accelerationStructureSize)
            .setType(type)
    );

    auto scratch_buf = VulkanBufferInfo{};
    TL_TRY_ASSIGN(scratch_buf, create_buffer(
        dev,
        VulkanBufferInfo::Type::Scratch,
        build_sizes.buildScratchSize,
        nullptr
    ));

    geom_build_info
        .setScratchData(scratch_buf.device_addr)
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
        .setPrimitiveCount(primitive_count)
        .setPrimitiveOffset(0)
        .setTransformOffset(0);
    
    TL_CHECK(do_work_now(dev, cmd_pool, [&](vk::CommandBuffer& cmd_buf) {
        cmd_buf.buildAccelerationStructuresKHR(geom_build_info, &build_range_info);
    }));

    auto desc_info = vk::WriteDescriptorSetAccelerationStructureKHR{ *accel };

    return VulkanAccelStructInfo {
        { std::move(accel) },
        std::move(accel_buf),
        std::move(geom_build_info),
        std::move(desc_info)
    };
}

[[nodiscard]] auto create_bottom_accel_for(
    VulkanDeviceInfo const& dev,
    VulkanCmdPoolInfo const& cmd_pool,
    Object const& obj
) -> tl::expected<VulkanBottomAccelStructInfo, std::string> {
    // note: transform will be later applied to the instances of the acceleration structure
    // so it's not needed here
    auto vert_buf = VulkanBufferInfo{};
    auto index_buf = VulkanBufferInfo{};

    auto vertices = std::vector<glm::vec3>{};
    vertices.reserve(obj.get_vertices().size());
    std::transform(
        obj.get_vertices().begin(), obj.get_vertices().end(),
        std::back_inserter(vertices),
        [](Vertex const& v) { return v.position; }
    );
    TL_TRY_ASSIGN(vert_buf, create_buffer(
        dev,
        VulkanBufferInfo::Type::AccelInput,
        sizeof(decltype(vertices)::value_type) * vertices.size(),
        vertices.data()
    ));
    TL_TRY_ASSIGN(index_buf, create_buffer(
        dev,
        VulkanBufferInfo::Type::AccelInput,
        sizeof(decltype(obj.get_indices())::value_type) * obj.get_indices().size(),
        obj.get_indices().data()
    ));
    auto prim_cnt = static_cast<uint32_t>(obj.get_indices().size() / 3);
    auto triangle_data = vk::AccelerationStructureGeometryTrianglesDataKHR{}
        .setVertexFormat(vk::Format::eR32G32B32Sfloat)
        .setVertexData(vert_buf.device_addr)
        .setVertexStride(sizeof(float) * 3)
        .setMaxVertex(prim_cnt)
        .setIndexType(vk::IndexType::eUint32)
        .setIndexData(index_buf.device_addr);
    auto geometry = vk::AccelerationStructureGeometryKHR{}
        .setGeometryType(vk::GeometryTypeKHR::eTriangles)
        .setFlags(vk::GeometryFlagBitsKHR::eOpaque)
        .setGeometry(vk::AccelerationStructureGeometryDataKHR{ triangle_data });
    auto build_info = vk::AccelerationStructureBuildGeometryInfoKHR{}
        .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace | 
            vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess)
        .setGeometryCount(1)
        .setPGeometries(&geometry)
        .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
        .setType(vk::AccelerationStructureTypeKHR::eBottomLevel);

    auto accel = VulkanAccelStructInfo{};
    TL_TRY_ASSIGN(accel,
        create_accel(dev, cmd_pool, build_info, vk::AccelerationStructureTypeKHR::eBottomLevel, prim_cnt)
    );
    
    return VulkanBottomAccelStructInfo{
        std::move(accel),
        std::move(vert_buf),
        std::move(index_buf)
    };
}

[[nodiscard]] auto create_top_accel_for(
    VulkanDeviceInfo const& dev,
    VulkanCmdPoolInfo const& cmd_pool,
    Scene const& scene
) -> tl::expected<VulkanTopAccelStructInfo, std::string> {
    auto accel_ins_vec = std::vector<vk::AccelerationStructureInstanceKHR>{};
    auto bottom_accels = std::vector<VulkanBottomAccelStructInfo>{};
    for(auto const& obj : scene.get_objects()) {
        auto accel = VulkanBottomAccelStructInfo{};
        TL_TRY_ASSIGN(accel, create_bottom_accel_for(dev, cmd_pool, obj));

        auto mat_data = glm::mat3x4 { obj.get_transform().get_matrix() };
        auto mat_data_arr = std::array {
            std::array { mat_data[0][0], mat_data[0][1], mat_data[0][2], mat_data[0][3] },
            std::array { mat_data[1][0], mat_data[1][1], mat_data[1][2], mat_data[1][3] },
            std::array { mat_data[2][0], mat_data[2][1], mat_data[2][2], mat_data[2][3] }
        };

        auto accel_inst_buf = VulkanBufferInfo{};
        auto accel_ins = vk::AccelerationStructureInstanceKHR{}
            .setTransform(vk::TransformMatrixKHR{ mat_data_arr })
            .setInstanceCustomIndex(0)
            .setMask(0xFF)
            .setInstanceShaderBindingTableRecordOffset(0)
            .setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable)
            .setAccelerationStructureReference(accel.accel.storage_mem.device_addr);
        accel_ins_vec.emplace_back(accel_ins);
        bottom_accels.emplace_back(std::move(accel));
    }
    auto accel_ins_buf = VulkanBufferInfo{};
    TL_TRY_ASSIGN(accel_ins_buf, create_buffer(
        dev,
        VulkanBufferInfo::Type::AccelInput,
        sizeof(decltype(accel_ins_vec)::value_type) * accel_ins_vec.size(),
        accel_ins_vec.data()
    ));
    auto prim_cnt = static_cast<uint32_t>(accel_ins_vec.size());
    auto instance_data = vk::AccelerationStructureGeometryInstancesDataKHR{}
        .setArrayOfPointers(false)
        .setData(accel_ins_buf.device_addr);
    auto geometry = vk::AccelerationStructureGeometryKHR{}
        .setGeometryType(vk::GeometryTypeKHR::eInstances)
        .setFlags(vk::GeometryFlagBitsKHR::eOpaque)
        .setGeometry(vk::AccelerationStructureGeometryDataKHR{ instance_data });
    auto build_info = vk::AccelerationStructureBuildGeometryInfoKHR{}
        .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace | 
            vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess)
        .setGeometryCount(1)
        .setPGeometries(&geometry)
        .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
        .setType(vk::AccelerationStructureTypeKHR::eTopLevel);
    auto accel = VulkanAccelStructInfo{};
    TL_TRY_ASSIGN(accel, create_accel(
        dev, cmd_pool,
        build_info, vk::AccelerationStructureTypeKHR::eTopLevel, prim_cnt)
    );
    return VulkanTopAccelStructInfo{
        std::move(accel),
        std::move(bottom_accels),
        std::move(accel_ins_vec)
    };
}

[[nodiscard]] auto create_rt_pipeline(
    VulkanDeviceInfo const& dev,
    VulkanCmdPoolInfo const& cmd_pool,
    VulkanImageInfo const& output_img,
    VulkanDescSetPoolInfo const& desc_set_pool,
    Scene const& scene
) -> tl::expected<VulkanPipelineInfo, std::string> {
    auto vk_top_accel = VulkanTopAccelStructInfo{};
    TL_TRY_ASSIGN(vk_top_accel, create_top_accel_for(dev, cmd_pool, scene));
    auto ray_gen_shader = VulkanShaderInfo{};
    auto miss_shader = VulkanShaderInfo{};
    auto chit_shader = VulkanShaderInfo{};

    auto shader_infos = std::array<VulkanShaderInfo, 3> {};
    TL_TRY_ASSIGN(shader_infos[0], create_shader_glsl(
        dev,
        R"(
            #version 460
            #extension GL_EXT_ray_tracing : enable
            layout (binding = 0) uniform accelerationStructureEXT topLevelAS;
            layout (binding = 1, rgba8) uniform image2D outputImage;

            void main() {
                imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(1.0, 0.0, 0.0, 1.0));
            }
        )",
        "ray_gen_shader",
        vk::ShaderStageFlagBits::eRaygenKHR
    ));
    TL_TRY_ASSIGN(shader_infos[1], create_shader_glsl(
        dev,
        R"(
            #version 460
            #extension GL_EXT_ray_tracing : enable
            void main() {
                return;
            }
        )",
        "miss_shader",
        vk::ShaderStageFlagBits::eMissKHR
    ));
    TL_TRY_ASSIGN(shader_infos[2], create_shader_glsl(
        dev,
        R"(
            #version 460
            #extension GL_EXT_ray_tracing : enable
            void main() {
                return;
            }
        )",
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
        }
    };
    auto shader_groups = std::array{
        vk::RayTracingShaderGroupCreateInfoKHR{
            vk::RayTracingShaderGroupTypeKHR::eGeneral,
            0,
            VK_SHADER_UNUSED_KHR,
            VK_SHADER_UNUSED_KHR,
            VK_SHADER_UNUSED_KHR
        },
        vk::RayTracingShaderGroupCreateInfoKHR{
            vk::RayTracingShaderGroupTypeKHR::eGeneral,
            1,
            VK_SHADER_UNUSED_KHR,
            VK_SHADER_UNUSED_KHR,
            VK_SHADER_UNUSED_KHR
        },
        vk::RayTracingShaderGroupCreateInfoKHR{
            vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
            2,
            VK_SHADER_UNUSED_KHR,
            VK_SHADER_UNUSED_KHR,
            VK_SHADER_UNUSED_KHR
        }
    };

    auto bindings = std::array{
        vk::DescriptorSetLayoutBinding {}
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR),
        vk::DescriptorSetLayoutBinding {}
            .setBinding(1)
            .setDescriptorType(vk::DescriptorType::eStorageImage)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR),
    };

    try {
        auto desc_set_layout = dev->createDescriptorSetLayoutUnique(
            vk::DescriptorSetLayoutCreateInfo{
                vk::DescriptorSetLayoutCreateFlags{},
                bindings
            }
        );
        auto pipeline_layout = dev->createPipelineLayoutUnique(
            vk::PipelineLayoutCreateInfo{}
            .setSetLayouts(*desc_set_layout)
            .setPushConstantRanges({})
        );
        auto pipeline = dev->createRayTracingPipelineKHRUnique(
            nullptr, nullptr,
            vk::RayTracingPipelineCreateInfoKHR{}
            .setStages(shader_stages)
            .setGroups(shader_groups)
            .setMaxPipelineRayRecursionDepth(4)
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
        TL_TRY_ASSIGN(raygen_buf, create_buffer(
            dev,
            VulkanBufferInfo::Type::ShaderBindingTable,
            handle_size,
            handles.data()
        ));
        auto miss_buf = VulkanBufferInfo{};
        TL_TRY_ASSIGN(miss_buf, create_buffer(
            dev,
            VulkanBufferInfo::Type::ShaderBindingTable,
            handle_size,
            handles.data() + handle_size_aligned
        ));
        auto hit_buf = VulkanBufferInfo{};
        TL_TRY_ASSIGN(hit_buf, create_buffer(
            dev,
            VulkanBufferInfo::Type::ShaderBindingTable,
            handle_size,
            handles.data() + handle_size_aligned * 2
        ));

        auto raygen_region = vk::StridedDeviceAddressRegionKHR{
            raygen_buf.device_addr,
            handle_size_aligned,
            handle_size_aligned
        };
        auto miss_region = vk::StridedDeviceAddressRegionKHR{
            miss_buf.device_addr,
            handle_size_aligned,
            handle_size_aligned
        };
        auto hit_region = vk::StridedDeviceAddressRegionKHR{
            hit_buf.device_addr,
            handle_size_aligned,
            handle_size_aligned
        };

        auto desc_set_layouts = std::array{ *desc_set_layout };
        auto desc_sets = dev->allocateDescriptorSetsUnique(
            vk::DescriptorSetAllocateInfo{
                *desc_set_pool,
                desc_set_layouts
            }
        );

        auto desc_writes = std::vector<vk::WriteDescriptorSet>(bindings.size());
        for (auto i = 0; i < bindings.size(); ++i) {
            desc_writes[i]
                .setDstSet(*desc_sets[0])
                .setDstBinding(bindings[i].binding)
                .setDescriptorCount(bindings[i].descriptorCount)
                .setDescriptorType(bindings[i].descriptorType);
        }
        desc_writes[0].setPNext(&vk_top_accel.accel.desc_info);
        desc_writes[1].setImageInfo(output_img.img_info);
        dev->updateDescriptorSets(desc_writes, {});

        return VulkanPipelineInfo{
            { std::move(pipeline.value) },
            std::move(pipeline_layout),
            std::move(desc_set_layout),
            std::move(desc_sets[0]),
            std::move(vk_top_accel),
            std::move(raygen_buf),
            std::move(miss_buf),
            std::move(hit_buf),
            raygen_region,
            miss_region,
            hit_region,
        };

    }
    catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

// a simple rasterization pipeline for testing
[[nodiscard]] auto test_create_render_pass() -> tl::expected<VulkanRenderPassInfo, std::string>;
[[nodiscard]] auto test_create_frame_buf() -> tl::expected<VulkanFrameBufferInfo, std::string>;
[[nodiscard]] auto test_create_pipeline() -> tl::expected<VulkanPipelineInfo, std::string>;


PTS::VulkanRayTracingRenderer::VulkanRayTracingRenderer(RenderConfig config)
	: Renderer{config, "Vulkan Ray Tracer"} {}

PTS::VulkanRayTracingRenderer::~VulkanRayTracingRenderer() noexcept {}


auto PTS::VulkanRayTracingRenderer::open_scene(View<Scene> scene) noexcept
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

    // (re)create pipeline
    TL_TRY_ASSIGN(m_vk_pipeline, create_rt_pipeline(m_vk_device, m_vk_cmd_pool, m_output_img,m_vk_desc_set_pool, scene));
    if (!m_vk_render_cmd_buf) {
        // create command buffer
       TL_TRY_ASSIGN(m_vk_render_cmd_buf, create_cmd_buf(m_vk_device, m_vk_cmd_pool, m_output_img, m_vk_pipeline));
    }
    // configure command buffer
    TL_CHECK_AND_PASS(config_cmd_buf(*m_vk_render_cmd_buf, m_vk_pipeline, m_output_img, m_config.width, m_config.height));

    return {};
}

auto PTS::VulkanRayTracingRenderer::on_add_editable(EditableView editable) noexcept
	-> tl::expected<void, std::string> {
	return {};
}

auto PTS::VulkanRayTracingRenderer::on_remove_editable(EditableView editable) noexcept
	-> tl::expected<void, std::string> {
	return {};
}

auto PTS::VulkanRayTracingRenderer::render(View<Camera> camera) noexcept
	-> tl::expected<void, std::string> {
	return {};
}

auto PTS::VulkanRayTracingRenderer::render_buffered(View<Camera> camera) noexcept
-> tl::expected<TextureHandle, std::string> {
    if (!m_vk_cmd_pool) {
        return TL_ERROR("command pool not initialized");
    }
    if (!m_vk_render_cmd_buf) {
        return TL_ERROR("command buffer not initialized");
    }
    m_vk_cmd_pool.queue.submit(
        vk::SubmitInfo{}
            .setCommandBuffers(*m_vk_render_cmd_buf),
        nullptr
    );
    m_vk_cmd_pool.queue.waitIdle();
    
    return m_output_img.img.gl_tex.get();
}

auto PTS::VulkanRayTracingRenderer::valid() const noexcept -> bool {
    return m_vk_ins && m_vk_device && m_vk_cmd_pool && m_vk_desc_set_pool && m_output_img.img.vk_image
        && m_output_img.img.gl_tex.get() && m_vk_pipeline && m_vk_render_cmd_buf;
}

auto PTS::VulkanRayTracingRenderer::on_change_render_config() noexcept -> tl::expected<void, std::string> {
    if (!m_vk_pipeline) {
        return TL_ERROR("pipeline not initialized");
    }
    if (!m_vk_render_cmd_buf) {
        return TL_ERROR("command buffer not initialized");
    }
    
	TL_CHECK_AND_PASS(config_cmd_buf(*m_vk_render_cmd_buf, m_vk_pipeline, m_output_img, m_config.width, m_config.height));
    return {};
}

auto PTS::VulkanRayTracingRenderer::init(ObserverPtr<Application> app) noexcept
	-> tl::expected<void, std::string> {
	TL_CHECK(Renderer::init(app));

    auto ins_ext = VulkanGLInteropUtils::get_vk_ins_exts();
    ins_ext.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

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
    device_ext.emplace_back(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME);

    using CreateInfoChainType = vk::StructureChain<
        vk::DeviceCreateInfo,
        vk::PhysicalDeviceBufferDeviceAddressFeatures,
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
        vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR
    >;
	TL_TRY_ASSIGN(m_vk_device, create_device(m_vk_ins, device_ext, std::function {
        [](vk::DeviceCreateInfo create_info) {
            return CreateInfoChainType { create_info, { true }, { true }, { true }, { true } };
        }
    }));

    TL_TRY_ASSIGN(m_vk_cmd_pool, create_cmd_pool(m_vk_device));
    TL_TRY_ASSIGN(m_vk_desc_set_pool, create_desc_set_pool(m_vk_device));
    TL_TRY_ASSIGN(m_output_img, create_tex(
        m_vk_device,
        vk::Format::eR8G8B8A8Unorm,
        k_max_width,
        k_max_height,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vk::ImageAspectFlagBits::eColor,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eGeneral,
        {}, 
        true
    ));

	return {};
}

auto PTS::VulkanRayTracingRenderer::draw_imgui() noexcept -> tl::expected<void, std::string> {
	return Renderer::draw_imgui();
}

#pragma region test
[[nodiscard]] auto test_create_render_pass(
    VulkanDeviceInfo const& dev
) -> tl::expected<VulkanRenderPassInfo, std::string> {
    try {
        auto const color_fmt = vk::Format::eR8G8B8A8Unorm;
        auto const depth_fmt = vk::Format::eD32Sfloat;
    
        auto props = dev.physical_device.getFormatProperties(color_fmt);
        if (!(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eColorAttachment)) {
            return TL_ERROR("color attachment not supported");
        }
        props = dev.physical_device.getFormatProperties(depth_fmt);
        if (!(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)) {
            return TL_ERROR("depth attachment not supported");
        }

        auto color_attch_desc = vk::AttachmentDescription{
            vk::AttachmentDescriptionFlags{},
            color_fmt,
            vk::SampleCountFlagBits::e1,
            vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferSrcOptimal
        };
        auto depth_attch_desc = vk::AttachmentDescription{
            vk::AttachmentDescriptionFlags{},
            depth_fmt,
            vk::SampleCountFlagBits::e1,
            vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eDontCare,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal
        };
        auto attch_descs = std::array {
            color_attch_desc,
            depth_attch_desc
        };

        auto color_attch_ref = vk::AttachmentReference{
            0, vk::ImageLayout::eColorAttachmentOptimal
        };
        auto depth_attch_ref = vk::AttachmentReference{
            1, vk::ImageLayout::eDepthStencilAttachmentOptimal
        };
        auto subpass_desc = vk::SubpassDescription{
            vk::SubpassDescriptionFlags{},
            vk::PipelineBindPoint::eGraphics,
            0,
            nullptr,
            1,
            &color_attch_ref,
            nullptr,
            &depth_attch_ref,
            0,
            nullptr
        };
        auto subpass_descs = std::array {
            subpass_desc
        };

        auto deps = std::array {
            vk::SubpassDependency{
                VK_SUBPASS_EXTERNAL,
                0,
                vk::PipelineStageFlagBits::eBottomOfPipe,
                vk::PipelineStageFlagBits::eColorAttachmentOutput,
                vk::AccessFlagBits::eMemoryRead,
                vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
                vk::DependencyFlagBits::eByRegion
            },
            vk::SubpassDependency{
                0, VK_SUBPASS_EXTERNAL,
                vk::PipelineStageFlagBits::eColorAttachmentOutput,
                vk::PipelineStageFlagBits::eBottomOfPipe,
                vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
                vk::AccessFlagBits::eMemoryRead,
                vk::DependencyFlagBits::eByRegion
            }
        };

        auto pass = dev->createRenderPassUnique(
            vk::RenderPassCreateInfo {
                vk::RenderPassCreateFlags{},
                attch_descs,
                subpass_descs,
                deps
            }
        );

        return VulkanRenderPassInfo{{std::move(pass)}, color_fmt, depth_fmt };
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

[[nodiscard]] auto test_create_frame_buf(
    VulkanDeviceInfo& dev,
    VulkanRenderPassInfo const& render_pass,
    RenderConfig const& config
) -> tl::expected<VulkanFrameBufferInfo, std::string> {
    try {
        auto color_tex = VulkanImageInfo{};
        TL_TRY_ASSIGN(color_tex, create_tex(
            dev,
            render_pass.color_fmt,
            config.width,
            config.height,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vk::ImageAspectFlagBits::eColor,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eGeneral,
            {},
            true
        ));
        auto depth_tex = VulkanImageInfo{};
        TL_TRY_ASSIGN(depth_tex, create_tex(
            dev,
            render_pass.depth_fmt,
            config.width,
            config.height,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vk::ImageAspectFlagBits::eDepth,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eGeneral,
            {},
            false
        ));

        auto attachments = std::array<vk::ImageView, 2> {
            color_tex.view.get(),
            depth_tex.view.get()
        };

        auto frame_buf = dev->createFramebufferUnique(
            vk::FramebufferCreateInfo{
                vk::FramebufferCreateFlags{},
                *render_pass,
                static_cast<uint32_t>(attachments.size()),
                attachments.data(),
                config.width,
                config.height,
                1
            }
        );
        return VulkanFrameBufferInfo{{std::move(frame_buf)}, std::move(color_tex), std::move(depth_tex) };
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

[[nodiscard]] auto test_create_pipeline(
    VulkanDeviceInfo const& dev,
    VulkanRenderPassInfo const& render_pass,
    VulkanFrameBufferInfo const& frame_buf,
    VulkanDescSetPoolInfo const& desc_set_pool,
    RenderConfig const& config
) -> tl::expected<VulkanPipelineInfo, std::string> {
    auto vertex_shader = VulkanShaderInfo{};
    TL_TRY_ASSIGN(vertex_shader, create_shader_glsl(
        dev,
        R"(
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(location = 0) in vec3 pos;
            layout(location = 1) in vec2 uv;
            layout(location = 0) out vec2 frag_uv;

            void main() {
                gl_Position = vec4(pos, 1.0);
                frag_uv = uv;
            }
        )",
        "vertex_shader",
        vk::ShaderStageFlagBits::eVertex
    ));
    auto fragment_shader = VulkanShaderInfo{};
    TL_TRY_ASSIGN(fragment_shader, create_shader_glsl(
        dev,
        R"(
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(location = 0) in vec2 frag_uv;
            layout(location = 0) out vec4 frag_color;
            void main() {
                frag_color = vec4(frag_uv, 0.0, 1.0);
            }
        )",
        "fragment_shader",
        vk::ShaderStageFlagBits::eFragment
    ));
    
    try {
        auto vert_stage_info = vk::PipelineShaderStageCreateInfo{
            vk::PipelineShaderStageCreateFlags{},
            vk::ShaderStageFlagBits::eVertex,
            *vertex_shader,
            "main",
            nullptr
        };
        auto frag_stage_info = vk::PipelineShaderStageCreateInfo{
            vk::PipelineShaderStageCreateFlags{},
            vk::ShaderStageFlagBits::eFragment,
            *fragment_shader,
            "main",
            nullptr
        };
        auto shader_stages = std::array {
            vert_stage_info,
            frag_stage_info
        };
        
        auto vert_binding_desc = std::array{
            vk::VertexInputBindingDescription{}
                .setBinding(0)
                .setStride(sizeof(float) * 5)
                .setInputRate(vk::VertexInputRate::eVertex),
        };
        auto vert_attrib_desc = std::array{
            vk::VertexInputAttributeDescription{}
                .setBinding(0)
                .setLocation(0)
                .setFormat(vk::Format::eR32G32B32Sfloat)
                .setOffset(0),
            vk::VertexInputAttributeDescription{}
                .setBinding(0)
                .setLocation(1)
                .setFormat(vk::Format::eR32G32Sfloat)
                .setOffset(sizeof(float) * 3)
        };
        auto vert_input_info = vk::PipelineVertexInputStateCreateInfo{
            vk::PipelineVertexInputStateCreateFlags{},
            vert_binding_desc,
            vert_attrib_desc
        };
        auto input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo{
            vk::PipelineInputAssemblyStateCreateFlags{},
            vk::PrimitiveTopology::eTriangleList,
            false
        };
        auto viewport = std::array {
                vk::Viewport{
                0.0f, 0.0f,
                static_cast<float>(config.width),
                static_cast<float>(config.height),
                0.0f, 1.0f
            }
        };
        auto scissor = std::array {
                vk::Rect2D{
                vk::Offset2D{ 0, 0 },
                vk::Extent2D{ config.width, config.height }
            }
        };
        auto viewport_state_info = vk::PipelineViewportStateCreateInfo{
            vk::PipelineViewportStateCreateFlags{},
            viewport,
            scissor
        };
        auto rasterizer_info = vk::PipelineRasterizationStateCreateInfo{
            vk::PipelineRasterizationStateCreateFlags{},
            false, false,
            vk::PolygonMode::eFill,
            vk::CullModeFlagBits::eNone,
            vk::FrontFace::eCounterClockwise,
            false,
            0.0f, 0.0f, 0.0f,
            1.0f
        };
        auto multisample_info = vk::PipelineMultisampleStateCreateInfo{
            vk::PipelineMultisampleStateCreateFlags{},
            vk::SampleCountFlagBits::e1,
            false,
            1.0f, nullptr,
            false, false
        };
        auto color_blend_attachment = std::array {
            vk::PipelineColorBlendAttachmentState{
                false,
                vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
                vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
                vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
            }
        };
        auto color_blend_info = vk::PipelineColorBlendStateCreateInfo{
            vk::PipelineColorBlendStateCreateFlags{},
            false,
            vk::LogicOp::eCopy,
            color_blend_attachment,
            { 0.0f, 0.0f, 0.0f, 0.0f }
        };
        auto depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo{
            vk::PipelineDepthStencilStateCreateFlags{},
            true, true,
            vk::CompareOp::eLess,
            false, false,
            vk::StencilOpState{},
            vk::StencilOpState{},
            0.0f, 1.0f
        };

        
        auto descriptor_set_layout_binding = std::array {
            vk::DescriptorSetLayoutBinding{
                0, vk::DescriptorType::eUniformBuffer,
                1, vk::ShaderStageFlagBits::eVertex,
                nullptr
            }
        };
        auto descriptor_set_layout = dev->createDescriptorSetLayoutUnique(
            vk::DescriptorSetLayoutCreateInfo{
                vk::DescriptorSetLayoutCreateFlags{},
                descriptor_set_layout_binding
            }
        );
        auto descriptor_set_layouts = std::array {
            *descriptor_set_layout
        };
        auto pipeline_layout = dev->createPipelineLayoutUnique(
            vk::PipelineLayoutCreateInfo{
                vk::PipelineLayoutCreateFlags{},
                descriptor_set_layouts,
                nullptr
            }
        );

        auto pipeline = dev->createGraphicsPipelineUnique(
            nullptr,
            vk::GraphicsPipelineCreateInfo{
                vk::PipelineCreateFlags{},
                shader_stages,
                &vert_input_info,
                &input_assembly_info,
                nullptr,
                &viewport_state_info,
                &rasterizer_info,
                &multisample_info,
                &depth_stencil_info,
                &color_blend_info,
                nullptr,
                *pipeline_layout,
                *render_pass,
                0,
                nullptr,
                0,
                nullptr
            }
        );
        if (pipeline.result != vk::Result::eSuccess) {
            return TL_ERROR("failed to create pipeline");
        }

        return VulkanPipelineInfo{{std::move(pipeline.value)}, std::move(pipeline_layout), std::move(descriptor_set_layout) };
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

#pragma endregion