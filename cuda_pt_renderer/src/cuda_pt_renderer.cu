#include "include/cuda_pt_renderer.cuh"

PTS::CUDA_PTRenderer::CUDA_PTRenderer(RenderConfig config, std::string_view name) noexcept
	: Renderer{config, name} {
	
}

PTS::CUDA_PTRenderer::~CUDA_PTRenderer() noexcept {
	
}

auto PTS::CUDA_PTRenderer::init(ObserverPtr<Application> app) noexcept -> tl::expected<void, std::string> {
	return Renderer::init(app);
}

auto PTS::CUDA_PTRenderer::draw_imgui() noexcept -> tl::expected<void, std::string> {
	return Renderer::draw_imgui();
}

auto PTS::CUDA_PTRenderer::open_scene(View<Scene> scene) noexcept -> tl::expected<void, std::string> {}

auto PTS::CUDA_PTRenderer::on_change_render_config(RenderConfig config) noexcept -> tl::expected<void, std::string> {}

auto PTS::CUDA_PTRenderer::on_add_editable(EditableView editable) noexcept -> tl::expected<void, std::string> {}

auto PTS::CUDA_PTRenderer::on_remove_editable(EditableView editable) noexcept -> tl::expected<void, std::string> {}

auto PTS::CUDA_PTRenderer::render(View<Camera> camera) noexcept -> tl::expected<void, std::string> {}

auto PTS::CUDA_PTRenderer::render_buffered(View<Camera> camera) noexcept -> tl::expected<TextureHandle, std::string> {}

auto PTS::CUDA_PTRenderer::valid() const noexcept -> bool {}

