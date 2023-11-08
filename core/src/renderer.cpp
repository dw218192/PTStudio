#include "include/renderer.h"
#include "application.h"

PTS::Renderer::Renderer(RenderConfig config, std::string_view name) noexcept : m_name(name),
                                                                               m_config(std::move(config)),
                                                                               m_app{nullptr} {}

PTS::Renderer::~Renderer() noexcept = default;

auto PTS::Renderer::init(ObserverPtr<Application> app) noexcept -> tl::expected<void, std::string> {
	m_app = app;
	return {};
}

auto PTS::Renderer::set_render_config(RenderConfig config) noexcept -> tl::expected<void, std::string> {
	if (!config.is_valid() || config == m_config) {
        return {};
    }
	m_config = config;
	return on_change_render_config();
}
