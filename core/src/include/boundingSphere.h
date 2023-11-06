#pragma once
#include <glm/glm.hpp>
#include "reflection.h"

struct BoundingSphere {
    BoundingSphere() noexcept = default;
    BoundingSphere(glm::vec3 center, float radius) noexcept : center{ center }, radius{ radius } {}

BEGIN_REFLECT(BoundingSphere);
    FIELD_MOD(glm::vec3, center, glm::vec3{0.0f},
        MSerialize{});
    FIELD_MOD(float, radius, 1.0f,
        MSerialize{});
END_REFLECT();
};