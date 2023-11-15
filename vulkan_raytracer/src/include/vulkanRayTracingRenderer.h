#pragma once

#include "renderer.h"
#include "glTexture.h"
#include "vulkanGLInterop.h"
#include <vulkan/vulkan.hpp>
#include <tcb/span.hpp>

namespace PTS {
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
    struct VulkanDescSetPoolInfo : VulkanInfo<vk::UniqueDescriptorPool> {};
    struct VulkanDescSetInfo : VulkanInfo<vk::UniqueDescriptorSet> {
        vk::UniqueDescriptorSetLayout layout {};
    };

    // if not shared, only vulkan part of this struct is valid
    struct VulkanImageInfo {
        VulkanGLInteropUtils::SharedImage img{};
        vk::UniqueImageView view {};
        vk::UniqueSampler sampler {};
        vk::DescriptorImageInfo img_info {};
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
    struct VulkanCmdBufInfo : VulkanInfo<vk::UniqueCommandBuffer> {};
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
    struct VulkanPipelineInfo : VulkanInfo<vk::UniquePipeline> {
        vk::UniquePipelineLayout layout{};
        VulkanTopAccelStructInfo top_accel{};

        // Desc Set related
        VulkanDescSetInfo per_scene_desc_set{};

        // Uniforms related
        VulkanBufferInfo camera_mem{};

        // SBT related
        VulkanBufferInfo raygen_mem{};
        VulkanBufferInfo miss_mem{};
        VulkanBufferInfo hit_mem{};
        vk::StridedDeviceAddressRegionKHR raygen_region{};
        vk::StridedDeviceAddressRegionKHR miss_region{};
        vk::StridedDeviceAddressRegionKHR hit_region{};
    };
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
		VulkanInsInfo m_vk_ins;
		VulkanDeviceInfo m_vk_device;
		VulkanCmdPoolInfo m_vk_cmd_pool;
        VulkanDescSetPoolInfo m_vk_desc_set_pool;
        VulkanImageInfo m_output_img; // image used for ray tracing output
        VulkanCmdBufInfo m_vk_render_cmd_buf;
        VulkanTopAccelStructInfo m_vk_top_accel;
        VulkanPipelineInfo m_vk_pipeline;
    };
}
