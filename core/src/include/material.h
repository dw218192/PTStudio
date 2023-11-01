#pragma once
#include <glm/glm.hpp>
#include "reflection.h"

struct Material {
    // metallic-roughness workflow
BEGIN_REFLECT(Material);
    FIELD_INIT(glm::vec3, albedo, 1.0f);
    FIELD_INIT(float, roughness, 0.5f);
    FIELD_INIT(float, metallic, 0.0f);
    FIELD_INIT(float, ior, 1.0f);
    FIELD_INIT(float, ao, 1.0f);
    FIELD_INIT(float, transmission, 0.0f);
END_REFLECT();
};