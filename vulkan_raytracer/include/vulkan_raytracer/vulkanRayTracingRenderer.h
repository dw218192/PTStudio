#pragma once

#include <core/legacy/continuousGPUBufferLink.h>
#include <core/legacy/renderer.h>

#include <optional>
#include <tcb/span.hpp>
#include <unordered_map>
#include <vulkan/vulkan.hpp>

// TODO: use pointer to impl to avoid exposing implementation details
#include "../../src/include/vulkanAccelStructInfo.h"
#include "../../src/include/vulkanHelper.h"
#include "../../src/include/vulkanRTPipelineInfo.h"

namespace PTS {
namespace Vk {
struct VulkanRayTracingRenderer final : Renderer {
    NO_COPY_MOVE(VulkanRayTracingRenderer);
    VulkanRayTracingRenderer(RenderConfig config);
    [[nodiscard]] auto open_scene(Ref<Scene> scene) noexcept
        -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render(View<Camera> camera) noexcept
        -> tl::expected<TextureHandle, std::string> override;
    [[nodiscard]] auto valid() const noexcept -> bool override;

   protected:
    [[nodiscard]] auto on_change_render_config() noexcept
        -> tl::expected<void, std::string> override;

   public:
    ~VulkanRayTracingRenderer() noexcept override;
    [[nodiscard]] auto init(ObserverPtr<Application> app) noexcept
        -> tl::expected<void, std::string> override;
    [[nodiscard]] auto draw_imgui() noexcept -> tl::expected<void, std::string> override;

   private:
    [[nodiscard]] auto reset_path_tracing() noexcept -> tl::expected<void, std::string>;

    // these funcs are called by on_add_obj & on_remove_obj or in the open_scene func
    [[nodiscard]] auto add_object(SceneObject const& obj) -> tl::expected<void, std::string>;
    [[nodiscard]] auto remove_object(SceneObject const& obj) -> tl::expected<void, std::string>;

    // callbacks
    [[nodiscard]] auto update_obj(SceneObject const& obj) noexcept
        -> tl::expected<void, std::string>;
    [[nodiscard]] auto update_light(Light const& light) noexcept -> tl::expected<void, std::string>;
    auto on_add_obj(Ref<SceneObject> obj) noexcept -> void;
    auto on_remove_obj(Ref<SceneObject> obj) noexcept -> void;
    Callback<void(Ref<SceneObject>)> m_on_add_obj{
        [this](Ref<SceneObject> data) { this->on_add_obj(data); }};
    Callback<void(Ref<SceneObject>)> m_on_remove_obj{
        [this](Ref<SceneObject> data) { this->on_remove_obj(data); }};
    DECL_FIELD_EVENT_MEMBERS(on_obj_world_trans_change, SceneObject,
                             SceneObject::FieldTag::WORLD_TRANSFORM);
    DECL_FIELD_EVENT_MEMBERS(on_obj_local_trans_change, SceneObject,
                             SceneObject::FieldTag::LOCAL_TRANSFORM);
    DECL_FIELD_EVENT_MEMBERS(on_mat_change, RenderableObject, RenderableObject::FieldTag::MAT);
    DECL_FIELD_EVENT_MEMBERS(on_light_type_change, Light, Light::FieldTag::LIGHT_TYPE);
    DECL_FIELD_EVENT_MEMBERS(on_light_color_change, Light, Light::FieldTag::COLOR);
    DECL_FIELD_EVENT_MEMBERS(on_light_intensity_change, Light, Light::FieldTag::INTENSITY);

    // fields
    ObserverPtr<Scene> m_scene{nullptr};
    VulkanInsInfo m_vk_ins;
    VulkanDeviceInfo m_vk_device;
    VulkanCmdPoolInfo m_vk_cmd_pool;
    VulkanDescSetPoolInfo m_vk_desc_set_pool;
    VulkanImageInfo m_output_img;  // image used for ray tracing output
    VulkanCmdBufInfo m_vk_render_cmd_buf;
    VulkanTopAccelStructInfo m_vk_top_accel;
    VulkanRTPipelineInfo m_vk_pipeline;

    ContinuousGPUBufferLink<LightData, Light, VulkanBufferInfo*, PTS::k_max_lights>
        m_light_data_link;

    // extra object data
    struct PerObjectData {
        size_t gpu_idx{};
    };

    std::unordered_map<ViewPtr<RenderableObject>, PerObjectData> m_rend_obj_data;

    struct EditingData {
        BEGIN_REFLECT(EditingData, void);
        FIELD(bool, unlimited_samples, false);

        FIELD(int, num_samples, 32, MRange{1, 1000});

        FIELD(int, max_bounces, 4, MRange{1, 100});

        FIELD(bool, direct_lighting_only, false);

        END_REFLECT();
    } m_editing_data;

    struct PathTracingData {
        int iteration{};
        std::optional<VulkanRayTracingShaders::CameraData>
            camera_data{};  // Already in PTS::Vk namespace
    } m_path_tracing_data;
};
}  // namespace Vk
}  // namespace PTS
