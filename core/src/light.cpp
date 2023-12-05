#include "light.h"
#include "scene.h"

PTS::Light::Light(ObjectConstructorUsage usage) noexcept
    : SceneObject{ usage } {}

PTS::Light::Light(Scene const& scene, Transform transform, glm::vec3 color, float intensity) noexcept
    : SceneObject{scene, transform}, m_color{ color }, m_intensity{ intensity } {}