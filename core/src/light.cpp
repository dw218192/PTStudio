#include "light.h"
#include "scene.h"

PTS::Light::Light(Scene const& scene, glm::vec3 color, float intensity, Transform transform) noexcept
    : m_name{scene.next_light_name()}, m_color{ color }, m_intensity{ intensity }, m_transform{ std::move(transform) }
{}

PTS::Light::Light(glm::vec3 color, float intensity, Transform transform) noexcept
    : m_color{color}, m_intensity{intensity}, m_transform{std::move(transform)}
{}

auto PTS::Light::get_name() const noexcept -> std::string_view {
    return m_name;
}

void PTS::Light::set_name(std::string_view name) noexcept {
    m_name = name;
}

auto PTS::Light::get_color() const noexcept -> glm::vec3 const& {
    return m_color;
}

auto PTS::Light::get_intensity() const noexcept -> float {
    return m_intensity;
}

auto PTS::Light::get_transform() const noexcept -> Transform const& {
    return m_transform;
}

void PTS::Light::set_color(glm::vec3 color) noexcept {
    m_color = color;
}

void PTS::Light::set_intensity(float intensity) noexcept {
    m_intensity = intensity;
}

void PTS::Light::set_transform(Transform transform) noexcept {
    m_transform = std::move(transform);
}
