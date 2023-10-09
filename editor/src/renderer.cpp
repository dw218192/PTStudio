#include "include/renderer.h"

Renderer::Renderer(RenderConfig const& config) noexcept :
    m_config(config),
    m_cam{ config.fovy, m_config.width / static_cast<float>(m_config.height), Transform {} }
{}

Renderer::~Renderer() noexcept = default;
