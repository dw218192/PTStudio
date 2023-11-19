#include "light.h"
#include "scene.h"

PTS::Light::Light(Scene const& scene, glm::vec3 color, float intensity, Transform transform) noexcept
    : m_name{scene.next_light_name()}, m_color{ color }, m_intensity{ intensity }, m_transform{ std::move(transform) }
{}

PTS::Light::Light(glm::vec3 color, float intensity, Transform transform) noexcept 
    : m_color{ color }, m_intensity{ intensity }, m_transform{ std::move(transform) }
{}