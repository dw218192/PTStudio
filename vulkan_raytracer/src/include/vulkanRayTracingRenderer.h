#pragma once

#include "renderer.h"
#include <vulkan/vulkan.hpp>

namespace PTS {
    struct VulkanRayTracingRenderer final : Renderer {
		NO_COPY_MOVE(VulkanRayTracingRenderer);
		VulkanRayTracingRenderer(RenderConfig config);
	    [[nodiscard]] auto open_scene(View<Scene> scene) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto on_add_editable(EditableView editable) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto on_remove_editable(EditableView editable) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto render(View<Camera> camera) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto render_buffered(View<Camera> camera) noexcept -> tl::expected<TextureHandle, std::string> override;
	    [[nodiscard]] auto valid() const noexcept -> bool override;

    protected:
	    [[nodiscard]] auto on_change_render_config() noexcept -> tl::expected<void, std::string> override;

    public:
	    ~VulkanRayTracingRenderer() noexcept override;
	    [[nodiscard]] auto init(ObserverPtr<Application> app) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto draw_imgui() noexcept -> tl::expected<void, std::string> override;

    private:
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
			std::vector<char const*> device_exts;
			std::vector<char const*> layers;
		};
		struct VulkanDeviceInfo : VulkanInfo<vk::UniqueDevice> {
            vk::PhysicalDevice physical_device;
			unsigned queue_family_idx{ 0 };
		};
		struct VulkanCmdPoolInfo : VulkanInfo<vk::UniqueCommandPool> {
			vk::Queue queue;
		};
        struct VulkanBufferInfo : VulkanInfo<vk::UniqueBuffer> {
            vk::UniqueDeviceMemory mem;
        };
        struct VulkanImageInfo : VulkanInfo<vk::UniqueImage> {
            vk::UniqueDeviceMemory mem;
            vk::UniqueImageView view;
            vk::Format format;
        };
        struct VulkanRenderPassInfo : VulkanInfo<vk::UniqueRenderPass> {
            vk::Format color_fmt;
            vk::Format depth_fmt;
        };
        struct VulkanFrameBufferInfo : VulkanInfo<vk::UniqueFramebuffer> {
            VulkanImageInfo color_tex;
            VulkanImageInfo depth_tex;
        };
        struct VulkanPipelineInfo : VulkanInfo<vk::UniquePipeline> {
            vk::UniquePipelineLayout layout;
        };
        
#define REQUIRES_INFO(...) // only for documentation
        [[nodiscard]] auto create_instance() -> tl::expected<VulkanInsInfo, std::string>;
        [[nodiscard]] auto create_device() -> tl::expected<VulkanDeviceInfo, std::string>;
        [[nodiscard]] auto create_cmd_pool() -> tl::expected<VulkanCmdPoolInfo, std::string>;
        [[nodiscard]] auto create_buffer(
            vk::BufferUsageFlags usage_flags,
            vk::MemoryPropertyFlags prop_flags, 
            vk::DeviceSize size,
            void* data
        ) -> tl::expected<VulkanBufferInfo, std::string>;
        
        [[nodiscard]] auto do_work_now(VulkanCmdPoolInfo const& cmd, vk::UniqueCommandBuffer cmd_buf)
            -> tl::expected<void, std::string>;
        
        [[nodiscard]] auto create_tex(
            vk::Format fmt,
            unsigned width, unsigned height,
            vk::ImageUsageFlags usage_flags,
            vk::MemoryPropertyFlags prop_flags,
            vk::ImageAspectFlags aspect_flags,
            vk::ImageLayout layout
        ) -> tl::expected<VulkanImageInfo, std::string>;

        [[nodiscard]] auto create_render_pass() -> tl::expected<VulkanRenderPassInfo, std::string>;
        [[nodiscard]] auto create_frame_buf() -> tl::expected<VulkanFrameBufferInfo, std::string>;
        [[nodiscard]] auto create_pipeline() -> tl::expected<VulkanPipelineInfo, std::string>;

		VulkanInsInfo m_vk_ins;
		VulkanDeviceInfo m_vk_device;
		VulkanCmdPoolInfo m_vk_cmd_pool;
        VulkanRenderPassInfo m_vk_render_pass;
        VulkanFrameBufferInfo m_vk_frame_buf;
    };
}
