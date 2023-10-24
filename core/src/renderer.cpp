#include "include/renderer.h"
#include "application.h"

Renderer::Renderer(RenderConfig config) noexcept : m_config(std::move(config)) {}
Renderer::~Renderer() noexcept = default;

auto Renderer::init(ObserverPtr<Application> app) noexcept -> tl::expected<void, std::string> {
	if (!app) {
		return TL_ERROR("application may not be null");
	}

	m_app = app;
	return {};
}