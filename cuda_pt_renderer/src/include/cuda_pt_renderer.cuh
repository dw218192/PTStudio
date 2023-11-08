#pragma once

#include "renderer.h"
#include "glTexture.h"

#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace PTS {
    struct CUDA_PTRenderer final : Renderer {
		NO_COPY_MOVE(CUDA_PTRenderer);

		explicit CUDA_PTRenderer(RenderConfig config) noexcept;
		~CUDA_PTRenderer() noexcept override;

	    [[nodiscard]] auto open_scene(View<Scene> scene) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto on_add_editable(EditableView editable) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto on_remove_editable(EditableView editable) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto render(View<Camera> camera) noexcept -> tl::expected<void, std::string> override;
	    [[nodiscard]] auto render_buffered(View<Camera> camera) noexcept -> tl::expected<TextureHandle, std::string> override;
	    [[nodiscard]] auto valid() const noexcept -> bool override;
		[[nodiscard]] auto init(ObserverPtr<Application> app) noexcept -> tl::expected<void, std::string> override;
		[[nodiscard]] auto draw_imgui() noexcept -> tl::expected<void, std::string> override;
	protected:
		[[nodiscard]] auto on_change_render_config() noexcept -> tl::expected<void, std::string> override;
	private:
		bool m_valid{ false };
		GLTextureRef m_render_tex{ nullptr };
		cudaGraphicsResource* m_cuda_image_res{ nullptr };
    };
};