#include "vulkanRayTracingRenderer.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <shaderc/shaderc.hpp>

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
namespace {
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
}

PTS::VulkanRayTracingRenderer::VulkanRayTracingRenderer(RenderConfig config)
	: Renderer{config, "Vulkan Ray Tracer"} {}

PTS::VulkanRayTracingRenderer::~VulkanRayTracingRenderer() noexcept {}


auto PTS::VulkanRayTracingRenderer::open_scene(View<Scene> scene) noexcept
	-> tl::expected<void, std::string> {
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
    TL_CHECK(do_work_now(m_vk_cmd_pool, *m_vk_render_cmd_buf));
    auto const& color_tex = m_vk_frame_buf.color_tex;
    return color_tex.img.gl_tex.get();
}

auto PTS::VulkanRayTracingRenderer::valid() const noexcept -> bool {
	return true;
}

auto PTS::VulkanRayTracingRenderer::on_change_render_config() noexcept
	-> tl::expected<void, std::string> {
	TL_CHECK_AND_PASS(config_cmd_buf(*m_vk_render_cmd_buf, m_config.width, m_config.height));
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
    device_ext.emplace_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    device_ext.emplace_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);

	TL_TRY_ASSIGN(m_vk_device, create_device(device_ext));
    TL_TRY_ASSIGN(m_vk_cmd_pool, create_cmd_pool());

    // create render pass
    TL_TRY_ASSIGN(m_vk_render_pass, create_render_pass());
    // create frame buffer
    TL_TRY_ASSIGN(m_vk_frame_buf, create_frame_buf());
    // create pipeline
    TL_TRY_ASSIGN(m_vk_pipeline, create_rt_pipeline());
    // create command buffer
    TL_TRY_ASSIGN(m_vk_render_cmd_buf, create_cmd_buf());

	return {};
}

auto PTS::VulkanRayTracingRenderer::draw_imgui() noexcept -> tl::expected<void, std::string> {
	return Renderer::draw_imgui();
}

auto PTS::VulkanRayTracingRenderer::create_instance(
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

auto PTS::VulkanRayTracingRenderer::create_device(tcb::span<std::string_view> required_device_ext) 
-> tl::expected<VulkanDeviceInfo, std::string> {
	if (!m_vk_ins) {
		return TL_ERROR("instance not created");
	}

	try {
		// get a physical device
		auto const physical_devices = m_vk_ins->enumeratePhysicalDevices();
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

		auto dev = physical_device.createDeviceUnique(
			vk::DeviceCreateInfo{
				vk::DeviceCreateFlags{},
				dev_create_infos,
				layers_cstr, // ignored, see https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDeviceCreateInfo.html
                dev_exts_cstr
            }
		);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*dev);

		return VulkanDeviceInfo{{std::move(dev)}, physical_device, queue_family_index };
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

auto PTS::VulkanRayTracingRenderer::create_cmd_pool() -> tl::expected<VulkanCmdPoolInfo, std::string> {
    if(!m_vk_device) {
        return TL_ERROR("device not created");
    }

	try {
		auto cmd_pool = m_vk_device->createCommandPoolUnique(
			vk::CommandPoolCreateInfo{
				vk::CommandPoolCreateFlags { 
                    vk::CommandPoolCreateFlagBits::eResetCommandBuffer
                },
				m_vk_device.queue_family_idx,
			}
		);
		auto const queue = m_vk_device->getQueue(m_vk_device.queue_family_idx, 0);

		return VulkanCmdPoolInfo{{std::move(cmd_pool)}, queue };
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

auto PTS::VulkanRayTracingRenderer::create_buffer(VulkanBufferInfo::Type type, vk::DeviceSize size, void* data)
-> tl::expected<VulkanBufferInfo, std::string> {
    if(!m_vk_device) {
        return TL_ERROR("device not created");
    }

    try {
        auto usage_flags = vk::BufferUsageFlags{};
        auto mem_props_flags = vk::MemoryPropertyFlags{};
        switch (type) {
        case VulkanBufferInfo::Type::Scratch:
            usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
            mem_props_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
            break;
        case VulkanBufferInfo::Type::AccelInput:
            usage_flags = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | 
                vk::BufferUsageFlagBits::eShaderDeviceAddress |
                vk::BufferUsageFlagBits::eStorageBuffer;
            mem_props_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
            break;
        case VulkanBufferInfo::Type::AccelStorage:
            usage_flags = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | 
                vk::BufferUsageFlagBits::eShaderDeviceAddress;
            mem_props_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
            break;
        case VulkanBufferInfo::Type::ShaderBindingTable:
            usage_flags = vk::BufferUsageFlagBits::eShaderBindingTableKHR | 
                vk::BufferUsageFlagBits::eShaderDeviceAddress;
            mem_props_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
            break;
        default:
            return TL_ERROR("invalid buffer type");
        }

        auto buffer = m_vk_device->createBufferUnique(
            vk::BufferCreateInfo{
                vk::BufferCreateFlags{},
                size,
                usage_flags,
                vk::SharingMode::eExclusive,
                0,
                nullptr
            }
        );

        // get property index
        auto const mem_req = m_vk_device->getBufferMemoryRequirements(*buffer);
        auto mem_type_idx = std::numeric_limits<uint32_t>::max();
        auto mem_props = m_vk_device.physical_device.getMemoryProperties();
        for (auto i = 0u; i < mem_props.memoryTypeCount; ++i) {
            if ((mem_req.memoryTypeBits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & mem_props_flags) == mem_props_flags) {
                mem_type_idx = i;
                break;
            }
        }

        auto mem = m_vk_device->allocateMemoryUnique(
            vk::MemoryAllocateInfo{
                mem_req.size,
                mem_type_idx        
            }
        );
        
        if (data) {
            auto const mapped = m_vk_device->mapMemory(*mem, 0, size);
            memcpy(mapped, data, size);
            m_vk_device->unmapMemory(*mem);
        }
        m_vk_device->bindBufferMemory(*buffer, *mem, 0);
        auto desc_info = vk::DescriptorBufferInfo{
            *buffer,
            0,
            size
        }; 
        auto device_addr = m_vk_device->getBufferAddressKHR(
            vk::BufferDeviceAddressInfo{
                *buffer
            }
        );
        return VulkanBufferInfo{{std::move(buffer)}, std::move(mem), std::move(desc_info), device_addr };
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

auto PTS::VulkanRayTracingRenderer::do_work_now(VulkanCmdPoolInfo const& cmd, vk::CommandBuffer const& cmd_buf)
-> tl::expected<void, std::string> {
    if(!m_vk_device) {
        return TL_ERROR("device not created");
    }

    try {
        auto const fence = m_vk_device->createFenceUnique(vk::FenceCreateInfo{});
        cmd.queue.submit(
            vk::SubmitInfo{
                0, nullptr, nullptr,
                1, &cmd_buf
            },
            *fence
        );

        auto const res = m_vk_device->waitForFences(*fence, true, std::numeric_limits<uint64_t>::max());
        if (res != vk::Result::eSuccess) {
            return TL_ERROR("failed to wait for fence");
        }

        m_vk_device->resetFences(*fence);
        return {};
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

auto PTS::VulkanRayTracingRenderer::create_tex(
    vk::Format fmt,
    unsigned width, unsigned height,
    vk::ImageUsageFlags usage_flags,
    vk::MemoryPropertyFlags prop_flags,
    vk::ImageAspectFlags aspect_flags,
    vk::ImageLayout layout,
    vk::SamplerCreateInfo sampler_info,
    bool shared
) -> tl::expected<VulkanImageInfo, std::string> {
	if(!m_vk_device) {
        return TL_ERROR("device not created");
    }

    try {
        auto img_info = vk::ImageCreateInfo{
            vk::ImageCreateFlags{},
            vk::ImageType::e2D,
            fmt,
            vk::Extent3D{ k_max_width, k_max_height, 1 },
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            usage_flags,
            vk::SharingMode::eExclusive,
            0,
            nullptr,
            layout
        };
        auto shared_img = VulkanGLInteropUtils::SharedImage{};

        if (shared) {
            TL_TRY_ASSIGN(shared_img,
                VulkanGLInteropUtils::create_shared_image(
                    *m_vk_device,
                    img_info,
                    prop_flags,
                    m_vk_device.physical_device.getMemoryProperties()
                )
            );
        } else {
            shared_img.vk_image = m_vk_device->createImageUnique(img_info);
            shared_img.mem.vk_mem = m_vk_device->allocateMemoryUnique(
                vk::MemoryAllocateInfo{
                    m_vk_device->getImageMemoryRequirements(*shared_img.vk_image).size,
                    m_vk_device.physical_device.getMemoryProperties().memoryTypes[0].heapIndex
                }
            );
        }
        // bind image to memory
        m_vk_device->bindImageMemory(*shared_img.vk_image, *shared_img.mem.vk_mem, 0);
        auto view = m_vk_device->createImageViewUnique(
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

        auto sampler = m_vk_device->createSamplerUnique(sampler_info);
        return VulkanImageInfo{
            std::move(shared_img),
            std::move(view),
            std::move(sampler)
        };

    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

auto PTS::VulkanRayTracingRenderer::create_render_pass() -> tl::expected<VulkanRenderPassInfo, std::string> {
    if (!m_vk_device) {
        return TL_ERROR("device not created");
    }

    try {
        auto const color_fmt = vk::Format::eR8G8B8A8Unorm;
        auto const depth_fmt = vk::Format::eD32Sfloat;
    
        auto props = m_vk_device.physical_device.getFormatProperties(color_fmt);
        if (!(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eColorAttachment)) {
            return TL_ERROR("color attachment not supported");
        }
        props = m_vk_device.physical_device.getFormatProperties(depth_fmt);
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

        auto pass = m_vk_device->createRenderPassUnique(
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


auto PTS::VulkanRayTracingRenderer::create_frame_buf() -> tl::expected<VulkanFrameBufferInfo, std::string> {
	if (!m_vk_device) {
        return TL_ERROR("device not created");
    }
    if (!m_vk_render_pass) {
        return TL_ERROR("render pass not created");
    }
    
    try {
        auto color_tex = VulkanImageInfo{};
        TL_TRY_ASSIGN(color_tex, create_tex(
            m_vk_render_pass.color_fmt,
            m_config.width,
            m_config.height,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vk::ImageAspectFlagBits::eColor,
            vk::ImageLayout::eUndefined,
            {},
            true
        ));
        auto depth_tex = VulkanImageInfo{};
        TL_TRY_ASSIGN(depth_tex, create_tex(
            m_vk_render_pass.depth_fmt,
            m_config.width,
            m_config.height,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vk::ImageAspectFlagBits::eDepth,
            vk::ImageLayout::eUndefined,
            {},
            false
        ));

        auto attachments = std::array<vk::ImageView, 2> {
            color_tex.view.get(),
            depth_tex.view.get()
        };

        auto frame_buf = m_vk_device->createFramebufferUnique(
            vk::FramebufferCreateInfo{
                vk::FramebufferCreateFlags{},
                *m_vk_render_pass,
                static_cast<uint32_t>(attachments.size()),
                attachments.data(),
                m_config.width,
                m_config.height,
                1
            }
        );
        return VulkanFrameBufferInfo{{std::move(frame_buf)}, std::move(color_tex), std::move(depth_tex) };
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

auto PTS::VulkanRayTracingRenderer::create_shader_glsl(std::string_view src, std::string_view name, vk::ShaderStageFlagBits stage)
-> tl::expected<VulkanShaderInfo, std::string> {
    if (!m_vk_device) {
        return TL_ERROR("device not created");
    }

    auto to_shaderc_stage = [](vk::ShaderStageFlagBits stage) {
        switch (stage) {
        case vk::ShaderStageFlagBits::eVertex:
            return shaderc_shader_kind::shaderc_vertex_shader;
        case vk::ShaderStageFlagBits::eFragment:
            return shaderc_shader_kind::shaderc_fragment_shader;
        default:
            return shaderc_shader_kind::shaderc_glsl_infer_from_source;
        }
    };

    try {
        auto compiler = shaderc::Compiler{};
        auto options = shaderc::CompileOptions{};
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
        auto shader = m_vk_device->createShaderModuleUnique(
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

auto PTS::VulkanRayTracingRenderer::create_test_pipeline() -> tl::expected<VulkanPipelineInfo, std::string> {
    if(!m_vk_device) {
        return TL_ERROR("device not created");
    }
    if(!m_vk_render_pass) {
        return TL_ERROR("render pass not created");
    }
    if(!m_vk_frame_buf) {
        return TL_ERROR("frame buffer not created");
    }

    auto vertex_shader = VulkanShaderInfo{};
    TL_TRY_ASSIGN(vertex_shader, create_shader_glsl(
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
                static_cast<float>(m_config.width),
                static_cast<float>(m_config.height),
                0.0f, 1.0f
            }
        };
        auto scissor = std::array {
                vk::Rect2D{
                vk::Offset2D{ 0, 0 },
                vk::Extent2D{ m_config.width, m_config.height }
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
        auto descriptor_set_layout = m_vk_device->createDescriptorSetLayoutUnique(
            vk::DescriptorSetLayoutCreateInfo{
                vk::DescriptorSetLayoutCreateFlags{},
                descriptor_set_layout_binding
            }
        );
        auto descriptor_set_layouts = std::array {
            *descriptor_set_layout
        };
        auto pipeline_layout = m_vk_device->createPipelineLayoutUnique(
            vk::PipelineLayoutCreateInfo{
                vk::PipelineLayoutCreateFlags{},
                descriptor_set_layouts,
                nullptr
            }
        );

        auto pipeline = m_vk_device->createGraphicsPipelineUnique(
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
                *m_vk_render_pass,
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

auto PTS::VulkanRayTracingRenderer::config_cmd_buf(vk::CommandBuffer& cmd_buf, unsigned width, unsigned height)
-> tl::expected<void, std::string> {
    try {
        cmd_buf.reset(vk::CommandBufferResetFlags{});
        cmd_buf.begin(vk::CommandBufferBeginInfo{});
        {
            vk::ClearValue clear_values[2];
            clear_values[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f } };
            clear_values[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

            cmd_buf.beginRenderPass(
                vk::RenderPassBeginInfo{
                    *m_vk_render_pass,
                    *m_vk_frame_buf,
                    vk::Rect2D{ vk::Offset2D{ 0, 0 }, vk::Extent2D{ width, height } },
                    static_cast<uint32_t>(std::size(clear_values)),
                    clear_values
                },
                vk::SubpassContents::eInline
            );
            {
                cmd_buf.setViewport(0, vk::Viewport{
                    0.0f, 0.0f,
                    static_cast<float>(width),
                    static_cast<float>(height),
                    0.0f, 1.0f
                });
                cmd_buf.setScissor(0, vk::Rect2D{
                    vk::Offset2D{ 0, 0 },
                    vk::Extent2D{ width, height }
                });

                cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_vk_pipeline);
                cmd_buf.bindVertexBuffers(0, *m_vk_vertex_buf, vk::DeviceSize{ 0 });
                cmd_buf.draw(6, 1, 0, 0);
            }
            cmd_buf.endRenderPass();
        }
        cmd_buf.end();
        return {};
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

auto PTS::VulkanRayTracingRenderer::create_cmd_buf() -> tl::expected<VulkanCmdBufInfo, std::string> {
    if (!m_vk_device) {
        return TL_ERROR("device not created");
    }
    if (!m_vk_cmd_pool) {
        return TL_ERROR("command pool not created");
    }
    if (!m_vk_frame_buf) {
        return TL_ERROR("frame buffer not created");
    }
    if (!m_vk_pipeline) {
        return TL_ERROR("pipeline not created");
    }

    auto cmd_buf = vk::UniqueCommandBuffer {};
    try {
        auto cmd_bufs = m_vk_device->allocateCommandBuffersUnique(
            vk::CommandBufferAllocateInfo{
                m_vk_cmd_pool.handle.get(),
                vk::CommandBufferLevel::ePrimary,
                1
            }
        );
        TL_CHECK(config_cmd_buf(cmd_bufs[0].get(), m_config.width, m_config.height));
        cmd_buf = std::move(cmd_bufs[0]);
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }

    return VulkanCmdBufInfo{{std::move(cmd_buf)}, vk::UniqueFence{} };
}

auto PTS::VulkanRayTracingRenderer::create_accel_struct() -> tl::expected<VulkanAccelStructInfo, std::string> {
	return {};
}

auto PTS::VulkanRayTracingRenderer::create_rt_pipeline() -> tl::expected<VulkanPipelineInfo, std::string> {
    return {};
}
