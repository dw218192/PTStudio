#include "include/renderer.h"

Renderer::Renderer(RenderConfig const& config) noexcept :
    m_config(config) {}

Renderer::~Renderer() noexcept = default;
