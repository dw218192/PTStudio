#pragma once
#include <glm/glm.hpp>
#include "reflection.h"

struct Material {
    // metallic-roughness workflow
BEGIN_REFLECT(Material);
	FIELD_MOD(glm::vec3, albedo, glm::vec3{1.0f},
        MSerialize{}, MColor{});
    FIELD_MOD(float, roughness, 0.5f,
        MSerialize{});
    FIELD_MOD(float, metallic, 0.0f,
        MSerialize{});
    FIELD_MOD(float, ior, 1.0f,
        MSerialize{});
    FIELD_MOD(float, ao, 1.0f,
        MSerialize{});
    FIELD_MOD(float, transmission, 0.0f,
        MSerialize{});
END_REFLECT();
};