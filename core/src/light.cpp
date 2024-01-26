#include "light.h"
#include "scene.h"

PTS::Light::Light(ObjectConstructorUsage usage) noexcept
	: SceneObject{usage} {}

PTS::Light::Light(Scene& scene, Transform transform, EditFlags edit_flags, LightType type, glm::vec3 color,
                  float intensity) noexcept
	: SceneObject{scene, std::move(transform), edit_flags}, m_type{type}, m_color{color}, m_intensity{intensity} {}

void PTS::Light::set_color(glm::vec3 color) noexcept {
	m_color = color;
	get_field_info<&Light::m_color>().on_change(m_color, *this);
}

void PTS::Light::set_intensity(float intensity) noexcept {
	m_intensity = intensity;
	get_field_info<&Light::m_intensity>().on_change(m_intensity, *this);
}
