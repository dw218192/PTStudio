#pragma once

#include "renderer.h"
#include "vulkanHelper.h"
#include "vulkanAccelStructInfo.h"
#include "vulkanRTPipelineInfo.h"
#include "vulkanRayTracingShaders.h"

#include <vulkan/vulkan.hpp>
#include <tcb/span.hpp>
#include <unordered_map>
#include <optional>

namespace PTS {
	struct VulkanRayTracingRenderer final : Renderer {
		NO_COPY_MOVE(VulkanRayTracingRenderer);
		VulkanRayTracingRenderer(RenderConfig config);
	    [[nodiscard]] auto open_scene(View<Scene> scene) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto on_add_obj(Ref<SceneObject> obj) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto on_remove_obj(Ref<SceneObject> obj) noexcept -> tl::expected<void, std::string> override;
        [[nodiscard]] auto on_obj_change(Ref<SceneObject> obj, SceneObjectChangeType type) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto render(View<Camera> camera) noexcept -> tl::expected<TextureHandle, std::string> override;
	    [[nodiscard]] auto valid() const noexcept -> bool override;

    protected:
	    [[nodiscard]] auto on_change_render_config() noexcept -> tl::expected<void, std::string> override;

    public:
	    ~VulkanRayTracingRenderer() noexcept override;
	    [[nodiscard]] auto init(ObserverPtr<Application> app) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto draw_imgui() noexcept -> tl::expected<void, std::string> override;

    private:
        [[nodiscard]] auto reset_path_tracing() noexcept -> tl::expected<void, std::string>;
        [[nodiscard]] auto add_object(RenderableObject const& obj) -> tl::expected<void, std::string>;
        [[nodiscard]] auto remove_object(RenderableObject const& obj) -> tl::expected<void, std::string>;

		VulkanInsInfo m_vk_ins;
		VulkanDeviceInfo m_vk_device;
		VulkanCmdPoolInfo m_vk_cmd_pool;
        VulkanDescSetPoolInfo m_vk_desc_set_pool;
        VulkanImageInfo m_output_img; // image used for ray tracing output
        VulkanCmdBufInfo m_vk_render_cmd_buf;
        VulkanTopAccelStructInfo m_vk_top_accel;
        VulkanRTPipelineInfo m_vk_pipeline;

        // extra object data
        struct PerObjectData {
            size_t gpu_idx{};
        };
        std::unordered_map<ViewPtr<RenderableObject>, PerObjectData> m_obj_data;

        struct EditingData {
            BEGIN_REFLECT(EditingData, void);
            FIELD(bool, unlimited_samples, false);
            FIELD(int, num_samples, 32,
                MRange { 1, 1000 });
            FIELD(int, max_bounces, 4,
                MRange { 1, 100 });
            END_REFLECT();
        } m_editing_data;

        struct PathTracingData {
            int iteration{};
            std::optional<CameraData> camera_data{};
        } m_path_tracing_data;
    };
}
