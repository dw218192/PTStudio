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
        [[nodiscard]] auto create_instance() -> tl::expected<vk::UniqueInstance, std::string>;
        [[nodiscard]] auto create_device() -> tl::expected<vk::UniqueDevice, std::string>;
		[[nodiscard]] auto create_frame_buffer() -> tl::expected<void, std::string>;

        // prepare shared texture between vulkan and opengl
        [[nodiscard]] auto prepared_shared_tex() -> tl::expected<void, std::string>;


        vk::UniqueInstance m_vk_ins;
		vk::UniqueDevice m_vk_device;

		struct {
			vk::UniqueImage image;
			vk::UniqueDeviceMemory mem;
			vk::UniqueImageView view;
		} m_vk_frame_buf;
    };
}
