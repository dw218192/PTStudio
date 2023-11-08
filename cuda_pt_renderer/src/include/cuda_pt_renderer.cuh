#pragma once

#include "renderer.h"
#include <thrust/device_vector.h>

namespace PTS {
    struct CUDA_PTRenderer final : Renderer {
		NO_COPY_MOVE(CUDA_PTRenderer);

		explicit CUDA_PTRenderer(RenderConfig config, std::string_view name) noexcept;
		~CUDA_PTRenderer() noexcept override;

	    [[nodiscard]] auto open_scene(View<Scene> scene) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto on_change_render_config(RenderConfig config) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto on_add_editable(EditableView editable) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto on_remove_editable(EditableView editable) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto render(View<Camera> camera) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto render_buffered(View<Camera> camera) noexcept -> tl::expected<TextureHandle, std::string> override;
	    [[nodiscard]] auto valid() const noexcept -> bool override;
		[[nodiscard]] auto init(ObserverPtr<Application> app) noexcept -> tl::expected<void, std::string> override;
		[[nodiscard]] auto draw_imgui() noexcept -> tl::expected<void, std::string> override;
    };
};