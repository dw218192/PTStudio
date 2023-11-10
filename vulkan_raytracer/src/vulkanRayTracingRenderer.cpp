#include "vulkanRayTracingRenderer.h"
#include <GLFW/glfw3.h>
#include <iostream>

static constexpr float k_quad_data_pos_uv[] = {
	// First triangle (positions)
	-1.0f, -1.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
	1.0f,  1.0f, 0.0f,
	// Second triangle (positions)
	-1.0f, -1.0f, 0.0f,
	1.0f,  1.0f, 0.0f,
	-1.0f,  1.0f, 0.0f,
	// First triangle (UVs)
	0.0f, 0.0f,
	1.0f, 0.0f,
	1.0f, 1.0f,
	// Second triangle (UVs)
	0.0f, 0.0f,
	1.0f, 1.0f,
	0.0f, 1.0f
};

static constexpr int k_max_width = 4000;
static constexpr int k_max_height = 4000;


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
	return nullptr;
}

auto PTS::VulkanRayTracingRenderer::valid() const noexcept -> bool {
	return true;
}

auto PTS::VulkanRayTracingRenderer::on_change_render_config() noexcept
	-> tl::expected<void, std::string> {
	return {};
}

auto PTS::VulkanRayTracingRenderer::init(ObserverPtr<Application> app) noexcept
	-> tl::expected<void, std::string> {
	TL_CHECK(Renderer::init(app));
	TL_TRY_ASSIGN(m_vk_ins, create_instance());
	TL_TRY_ASSIGN(m_vk_device, create_device());
    TL_TRY_ASSIGN(m_vk_cmd_pool, create_cmd_pool());
    
    // upload vertex data to VRAM
    {
        VulkanBufferInfo staging_buffer;
        TL_TRY_ASSIGN(staging_buffer, create_buffer(
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            sizeof(k_quad_data_pos_uv),
            const_cast<void*>(static_cast<void const*>(k_quad_data_pos_uv))
        ));
        VulkanBufferInfo vertex_buffer;
        TL_TRY_ASSIGN(vertex_buffer, create_buffer(
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            sizeof(k_quad_data_pos_uv),
            nullptr
        ));

        auto copy_cmd = vk::UniqueCommandBuffer {};
        try {
            auto cmds = m_vk_device->allocateCommandBuffersUnique(
                vk::CommandBufferAllocateInfo{
                    m_vk_cmd_pool.handle.get(),
                    vk::CommandBufferLevel::ePrimary,
                    1
                }
            );
            cmds[0]->begin(vk::CommandBufferBeginInfo{});
            cmds[0]->copyBuffer(staging_buffer.handle.get(), vertex_buffer.handle.get(), vk::BufferCopy {
                0, 0, sizeof(k_quad_data_pos_uv)
            });
            cmds[0]->end();
            copy_cmd = std::move(cmds[0]);
        } catch (vk::SystemError& err) {
            return TL_ERROR(err.what());
        }

        TL_CHECK_AND_PASS(do_work_now(m_vk_cmd_pool, std::move(copy_cmd)));
    }

    // create render pass
    TL_TRY_ASSIGN(m_vk_render_pass, create_render_pass());
    // create frame buffer
    TL_TRY_ASSIGN(m_vk_frame_buf, create_frame_buf());

	return {};
}

auto PTS::VulkanRayTracingRenderer::draw_imgui() noexcept -> tl::expected<void, std::string> {
	return Renderer::draw_imgui();
}

auto PTS::VulkanRayTracingRenderer::create_instance() -> tl::expected<VulkanInsInfo, std::string> {
	auto glfw_extensions_count = 0u;
	auto glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extensions_count);
	auto device_exts = std::vector<const char*>{
		glfw_extensions,
		glfw_extensions + glfw_extensions_count
	};
	device_exts.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

	auto const app_info = vk::ApplicationInfo{
		"PTS::VulkanRayTracingRenderer",
		VK_MAKE_VERSION(1, 0, 0),
		"PTS",
		VK_MAKE_VERSION(1, 0, 0),
		VK_API_VERSION_1_2
	};

	try {
		// check if all extensions are available
		auto ext_props = vk::enumerateInstanceExtensionProperties();
		for (auto& ext : device_exts) {
			auto found = false;
			for (auto& prop : ext_props) {
				if (strcmp(ext, prop.extensionName) == 0) {
					found = true;
					break;
				}
			}
			if (!found) {
				return TL_ERROR("extension not found: " + std::string{ext});
			}
		}

		// check if all layers are available
		auto layers = std::vector<const char*>{};
#ifndef NDEBUG
		layers.push_back("VK_LAYER_KHRONOS_validation");
#endif
		auto layer_props = vk::enumerateInstanceLayerProperties();
		for (auto const& layer : layers) {
			auto found = false;
			for (auto& prop : layer_props) {
				if (strcmp(layer, prop.layerName) == 0) {
					found = true;
					break;
				}
			}
			if (!found) {
				return TL_ERROR("layer not found: " + std::string{layer});
			}
		}


		auto ins = vk::createInstanceUnique(
			vk::InstanceCreateInfo{
				vk::InstanceCreateFlags{
					VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
				},
				&app_info,
				static_cast<uint32_t>(layers.size()), layers.data(),
				static_cast<uint32_t>(device_exts.size()), device_exts.data()
			}
		);

		return VulkanInsInfo{{std::move(ins)}, device_exts, layers };
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

auto PTS::VulkanRayTracingRenderer::create_device() -> tl::expected<VulkanDeviceInfo, std::string> {
	if (!m_vk_ins) {
		return TL_ERROR("instance not created");
	}

	try {
		// get a physical device
		auto const physical_devices = m_vk_ins->enumeratePhysicalDevices();
		if (physical_devices.empty()) {
			return TL_ERROR("no physical device found");
		}
		auto const physical_device = physical_devices[0];

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
		auto const queue_priorities = 0.0f;
		auto const queue_create_info = vk::DeviceQueueCreateInfo{
			vk::DeviceQueueCreateFlags{},
			queue_family_index,
			1,
			&queue_priorities
		};

		// create logical device
		auto dev = physical_device.createDeviceUnique(
			vk::DeviceCreateInfo{
				vk::DeviceCreateFlags{},
				1,
				&queue_create_info,
				static_cast<uint32_t>(m_vk_ins.layers.size()),
				m_vk_ins.layers.data()
			}
		);

		return VulkanDeviceInfo{{std::move(dev)}, std::move(physical_device), queue_family_index };
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
				vk::CommandPoolCreateFlags {},
				m_vk_device.queue_family_idx,
			}
		);
		auto const queue = m_vk_device->getQueue(m_vk_device.queue_family_idx, 0);

		return VulkanCmdPoolInfo{{std::move(cmd_pool)}, queue };
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

auto PTS::VulkanRayTracingRenderer::create_buffer(vk::BufferUsageFlags usage_flags, vk::MemoryPropertyFlags prop_flags,
	vk::DeviceSize size, void* data) -> tl::expected<VulkanBufferInfo, std::string> {
    if(!m_vk_device) {
        return TL_ERROR("device not created");
    }

    try {
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
        auto const mem_props = m_vk_device.physical_device.getMemoryProperties();
        auto mem_type_idx = std::numeric_limits<uint32_t>::max();
        for (auto i = 0u; i < mem_props.memoryTypeCount; ++i) {
            if ((mem_req.memoryTypeBits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & prop_flags) == prop_flags) {
                mem_type_idx = i;
                break;
            }
        }
        if (mem_type_idx == std::numeric_limits<uint32_t>::max()) {
            return TL_ERROR("no suitable memory type found");
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
        return VulkanBufferInfo{{std::move(buffer)}, std::move(mem) };
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

auto PTS::VulkanRayTracingRenderer::do_work_now(VulkanCmdPoolInfo const& cmd, vk::UniqueCommandBuffer cmd_buf)
-> tl::expected<void, std::string> {
    if(!m_vk_device) {
        return TL_ERROR("device not created");
    }

    try {
        auto const fence = m_vk_device->createFenceUnique(vk::FenceCreateInfo{});
        cmd.queue.submit(
            vk::SubmitInfo{
                0, nullptr, nullptr,
                1, &cmd_buf.get()
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

auto PTS::VulkanRayTracingRenderer::create_tex(vk::Format fmt, unsigned width, unsigned height, vk::ImageUsageFlags usage_flags,
vk::MemoryPropertyFlags prop_flags, vk::ImageAspectFlags aspect_flags, vk::ImageLayout layout) -> tl::expected<VulkanImageInfo, std::string> 
{
	if(!m_vk_device) {
        return TL_ERROR("device not created");
    }

    try {
        auto img = m_vk_device->createImageUnique(
            vk::ImageCreateInfo{
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
                layout
            }
        );
        auto const mem_req = m_vk_device->getImageMemoryRequirements(*img);
        auto mem = m_vk_device->allocateMemoryUnique(
            vk::MemoryAllocateInfo{
                mem_req.size,
                m_vk_device.queue_family_idx
            }
        );
        m_vk_device->bindImageMemory(*img, *mem, 0);
        auto view = m_vk_device->createImageViewUnique(
            vk::ImageViewCreateInfo{
                vk::ImageViewCreateFlags{},
                *img,
                vk::ImageViewType::e2D,
                fmt,
                vk::ComponentMapping{},
                vk::ImageSubresourceRange{
                    aspect_flags,
                    0, 1, 0, 1
                }
            }
        );
        return VulkanImageInfo{{std::move(img)}, std::move(mem), std::move(view) };
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
        auto attch_descs = std::array<vk::AttachmentDescription, 2> {
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
        auto subpass_descs = std::array<vk::SubpassDescription, 1> {
            subpass_desc
        };

        auto deps = std::array<vk::SubpassDependency, 2> {
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
                static_cast<uint32_t>(attch_descs.size()),
                attch_descs.data(),
                static_cast<uint32_t>(subpass_descs.size()),
                subpass_descs.data(),
                static_cast<uint32_t>(deps.size()),
                deps.data()
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
            vk::ImageLayout::eUndefined
        ));
        auto depth_tex = VulkanImageInfo{};
        TL_TRY_ASSIGN(depth_tex, create_tex(
            m_vk_render_pass.depth_fmt,
            m_config.width,
            m_config.height,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vk::ImageAspectFlagBits::eDepth,
            vk::ImageLayout::eUndefined
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

    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

auto PTS::VulkanRayTracingRenderer::create_pipeline() -> tl::expected<VulkanPipelineInfo, std::string> {
    return {};
}
