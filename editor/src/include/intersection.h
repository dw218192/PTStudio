#pragma once

#include <glm/glm.hpp>
#include "ray.h"
#include "boundingBox.h"

namespace Intersection {
    struct Result {
        bool hit;
        float t;

        Result(bool hit) noexcept : hit{ hit }, t{ 0.0f } { }
        Result(bool hit, float t) noexcept : hit{ hit }, t{ t } { }
        operator bool() const noexcept { return hit; }
    };

    auto ray_box(BoundingBox const& box, Ray const& r) -> Result;
}