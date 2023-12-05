#pragma once
#include <glm/glm.hpp>
#include "reflection.h"
namespace PTS {
    struct BoundingSphere {
        BoundingSphere() noexcept = default;
        BoundingSphere(glm::vec3 center, float radius) noexcept : center{ center }, radius{ radius } {}

        BEGIN_REFLECT(BoundingSphere, void);
        FIELD(glm::vec3, center, glm::vec3{ 0.0f },
            MSerialize{});
        FIELD(float, radius, 1.0f,
            MSerialize{});
        END_REFLECT();
    };
}