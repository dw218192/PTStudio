#include "vulkanRayTracingRenderer.h"
#include <GLFW/glfw3.h>
#include <iostream>

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT severity,
	VkDebugUtilsMessageTypeFlagsEXT type,
	const VkDebugUtilsMessengerCallbackDataEXT* data,
	void* user_data
) {
	std::cerr << "validation layer: " << data->pMessage << std::endl;
	return VK_FALSE;
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
	return {};
}

auto PTS::VulkanRayTracingRenderer::draw_imgui() noexcept -> tl::expected<void, std::string> {
	return Renderer::draw_imgui();
}

auto PTS::VulkanRayTracingRenderer::create_instance() -> tl::expected<vk::UniqueInstance, std::string> {
	auto glfw_extensions_count = 0u;
	auto glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extensions_count);
	auto glfw_exts_vec = std::vector<const char*>{
		glfw_extensions,
		glfw_extensions + glfw_extensions_count
	};
	glfw_exts_vec.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
	auto app_info = vk::ApplicationInfo{
		"PTS::VulkanRayTracingRenderer", VK_MAKE_VERSION(1, 0, 0), "PTS", VK_MAKE_VERSION(1, 0, 0),
		VK_API_VERSION_1_2
	};
	try {
		// check if all extensions are available
		auto ext_props = vk::enumerateInstanceExtensionProperties();
		for (auto& ext : glfw_exts_vec) {
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
		for (auto& layer : layers) {
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


		auto ret = vk::createInstanceUnique(
			vk::InstanceCreateInfo{
				vk::InstanceCreateFlags{
					VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
				},
				&app_info,
				static_cast<uint32_t>(layers.size()), layers.data(),
				static_cast<uint32_t>(glfw_exts_vec.size()), glfw_exts_vec.data()
			}
		);
		return ret;
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

auto PTS::VulkanRayTracingRenderer::create_device() -> tl::expected<vk::UniqueDevice, std::string> {
	if (!m_vk_ins) {
		return TL_ERROR("instance not created");
	}

	try {
		// get a physical device
		auto physical_devices = m_vk_ins->enumeratePhysicalDevices();
		if (physical_devices.empty()) {
			return TL_ERROR("no physical device found");
		}
		auto physical_device = physical_devices[0];

		// request a single graphics queue
		auto queue_family_props = physical_device.getQueueFamilyProperties();
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
		auto queue_priorities = 0.0f;
		auto queue_create_info = vk::DeviceQueueCreateInfo{
			vk::DeviceQueueCreateFlags{},
			queue_family_index,
			1,
			&queue_priorities
		};

		// create logical device
		auto ret = physical_device.createDeviceUnique(
			vk::DeviceCreateInfo{
				vk::DeviceCreateFlags{},
				1,
				&queue_create_info,
				0,
				nullptr
			}
		);

		return ret;
	} catch (vk::SystemError& err) {
		return TL_ERROR(err.what());
	}
}

auto PTS::VulkanRayTracingRenderer::create_frame_buffer() -> tl::expected<void, std::string> {
	return {};
}
