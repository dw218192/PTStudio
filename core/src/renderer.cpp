#include "include/renderer.h"
#include "application.h"

Renderer::Renderer(RenderConfig config, std::string_view name) noexcept : m_name(name), m_config(std::move(config)) {}
Renderer::~Renderer() noexcept = default;

auto Renderer::init(ObserverPtr<Application> app) noexcept -> tl::expected<void, std::string> {
	m_app = app;
	return {};
}