#include "light.h"

auto Light::get_color() const noexcept -> glm::vec3 const& {
    return m_color;
}

auto Light::get_intensity() const noexcept -> float {
    return m_intensity;
}

auto Light::get_transform() const noexcept -> Transform const& {
    return m_transform;
}

void Light::set_color(glm::vec3 color) noexcept {
    m_color = color;
}

void Light::set_intensity(float intensity) noexcept {
    m_intensity = intensity;
}

void Light::set_transform(Transform transform) noexcept {
    m_transform = std::move(transform);
}
