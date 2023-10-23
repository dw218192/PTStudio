#include "include/renderer.h"

Renderer::Renderer(RenderConfig config) noexcept : m_config(std::move(config)) {}
Renderer::~Renderer() noexcept = default;