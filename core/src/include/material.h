#pragma once
#include <glm/glm.hpp>
#include "reflection.h"

struct Material {
    // metallic-roughness workflow
BEGIN_REFLECT(Material);
	FIELD_MOD(glm::vec3, albedo,
        MDefault{ glm::vec3 {1.0f} },
        MSerialize{});
    FIELD_MOD(float, roughness,
        MDefault{ 0.5f },
        MSerialize{});
    FIELD_MOD(float, metallic,
        MDefault{ 0.0f },
        MSerialize{});
    FIELD_MOD(float, ior,
        MDefault{ 1.0f },
        MSerialize{});
    FIELD_MOD(float, ao,
        MDefault{ 1.0f },
        MSerialize{});
    FIELD_MOD(float, transmission,
        MDefault{ 0.0f },
        MSerialize{});
END_REFLECT();
};