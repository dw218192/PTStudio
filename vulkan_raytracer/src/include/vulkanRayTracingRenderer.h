#pragma once

#include "renderer.h"
#include "glTexture.h"
#include "vulkanGLInterop.h"
#include <vulkan/vulkan.hpp>
#include <tcb/span.hpp>

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
        /**
         * @brief A wrapper for vulkan unique handles; used to logically bundle a handle with other related data
         * @tparam UniqueHandle vulkan unique handle type
        */
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
			std::vector<std::string_view> ins_exts {};
			std::vector<std::string_view> layers {};
		};
		struct VulkanDeviceInfo : VulkanInfo<vk::UniqueDevice> {
            vk::PhysicalDevice physical_device {};
			unsigned queue_family_idx{ 0 };
		};
		struct VulkanCmdPoolInfo : VulkanInfo<vk::UniqueCommandPool> {
			vk::Queue queue {};
		};
        // if not shared, only vulkan part of this struct is valid
        struct VulkanImageInfo {
            VulkanGLInteropUtils::SharedImage img{};
            vk::UniqueImageView view {};
            vk::UniqueSampler sampler {};
        };
        struct VulkanRenderPassInfo : VulkanInfo<vk::UniqueRenderPass> {
            vk::Format color_fmt {};
            vk::Format depth_fmt {};
        };
        struct VulkanFrameBufferInfo : VulkanInfo<vk::UniqueFramebuffer> {
            VulkanImageInfo color_tex {};
            VulkanImageInfo depth_tex {};
        };
        struct VulkanShaderInfo : VulkanInfo<vk::UniqueShaderModule> {
            vk::ShaderStageFlagBits stage {};
        };
        struct VulkanPipelineInfo : VulkanInfo<vk::UniquePipeline> {
            vk::UniquePipelineLayout layout {};
            vk::UniqueDescriptorSetLayout desc_set_layout {};
        };
        struct VulkanCmdBufInfo : VulkanInfo<vk::UniqueCommandBuffer> {
            vk::UniqueFence fence {};
        };
        struct VulkanBufferInfo : VulkanInfo<vk::UniqueBuffer> {
            enum class Type {
                Scratch,
                AccelInput,
                AccelStorage,
                ShaderBindingTable,
            };
            vk::UniqueDeviceMemory mem {};
            vk::DescriptorBufferInfo desc_info {};
            vk::DeviceAddress device_addr {};
        };
        struct VulkanAccelStructInfo : VulkanInfo<vk::UniqueAccelerationStructureKHR> {
            VulkanBufferInfo storage_mem {};
            vk::AccelerationStructureBuildGeometryInfoKHR geom_build_info {};
            vk::WriteDescriptorSetAccelerationStructureKHR desc_info {};
        };
        struct VulkanBottomAccelStructInfo {
            VulkanAccelStructInfo accel {};
            VulkanBufferInfo vertex_mem {};
            VulkanBufferInfo index_mem {};
        };
        struct VulkanTopAccelStructInfo {
            VulkanAccelStructInfo accel {};
            std::vector<VulkanBottomAccelStructInfo> bottom_accels {};
            std::vector<vk::AccelerationStructureInstanceKHR> instances {};
        };

        [[nodiscard]] auto create_instance(
            tcb::span<std::string_view> required_ins_ext,
            tcb::span<std::string_view> required_gl_ext
        ) -> tl::expected<VulkanInsInfo, std::string>;

        [[nodiscard]] auto create_device(tcb::span<std::string_view> required_device_ext)
          -> tl::expected<VulkanDeviceInfo, std::string>;

        [[nodiscard]] auto create_cmd_pool() -> tl::expected<VulkanCmdPoolInfo, std::string>;

        /**
         * @brief Convenience function to create a vulkan buffer
         * @param type the type of the buffer
         * @param size the size of the buffer, in bytes
         * @param data the data to be copied to the buffer, can be nullptr if the buffer is not to be initialized
        */
        [[nodiscard]] auto create_buffer(
            VulkanBufferInfo::Type type,
            vk::DeviceSize size,
            void const* data
        ) -> tl::expected<VulkanBufferInfo, std::string>;
        
        [[nodiscard]] auto do_work_now(vk::CommandBuffer const& cmd_buf) -> tl::expected<void, std::string>;
        
        [[nodiscard]] auto create_tex(
            vk::Format fmt,
            unsigned width, unsigned height,
            vk::ImageUsageFlags usage_flags,
            vk::MemoryPropertyFlags prop_flags,
            vk::ImageAspectFlags aspect_flags,
            vk::ImageLayout layout,
            vk::SamplerCreateInfo sampler_info,
            bool shared
        ) -> tl::expected<VulkanImageInfo, std::string>;

        [[nodiscard]] auto create_render_pass() -> tl::expected<VulkanRenderPassInfo, std::string>;
        [[nodiscard]] auto create_frame_buf() -> tl::expected<VulkanFrameBufferInfo, std::string>;
        [[nodiscard]] auto create_shader_glsl(std::string_view src, std::string_view name, vk::ShaderStageFlagBits stage)
            -> tl::expected<VulkanShaderInfo, std::string>;
        [[nodiscard]] auto config_cmd_buf(
            vk::CommandBuffer& cmd_buf,
            unsigned width, unsigned height
        ) -> tl::expected<void, std::string>;
        [[nodiscard]] auto create_cmd_buf() -> tl::expected<VulkanCmdBufInfo, std::string>;

        [[nodiscard]] auto create_accel(
            vk::AccelerationStructureBuildGeometryInfoKHR geom_build_info,
            vk::AccelerationStructureTypeKHR type,
            uint32_t primitive_count
        ) -> tl::expected<VulkanAccelStructInfo, std::string>;
        [[nodiscard]] auto create_bottom_accel_for(Object const& obj) -> tl::expected<VulkanBottomAccelStructInfo, std::string>;
        [[nodiscard]] auto create_top_accel_for(Scene const& scene) -> tl::expected<VulkanTopAccelStructInfo, std::string>;
        [[nodiscard]] auto create_rt_pipeline(Scene const& scene) -> tl::expected<VulkanPipelineInfo, std::string>;

        // a simple rasterization pipeline for testing
        [[nodiscard]] auto test_create_pipeline() -> tl::expected<VulkanPipelineInfo, std::string>;

		VulkanInsInfo m_vk_ins;
		VulkanDeviceInfo m_vk_device;
		VulkanCmdPoolInfo m_vk_cmd_pool;
        VulkanCmdBufInfo m_vk_render_cmd_buf;
        VulkanRenderPassInfo m_vk_render_pass;
        VulkanFrameBufferInfo m_vk_frame_buf;
        VulkanTopAccelStructInfo m_vk_top_accel;
        VulkanPipelineInfo m_vk_pipeline;
    };
}
