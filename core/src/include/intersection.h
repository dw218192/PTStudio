#pragma once

#include <glm/glm.hpp>
#include "ray.h"
#include "boundingBox.h"
#include "tcb/span.hpp"

namespace Intersection {
    struct Result {
        bool hit;
        float t;

        Result(bool hit) noexcept : hit{ hit }, t{ 0.0f } { }
        Result(bool hit, float t) noexcept : hit{ hit }, t{ t } { }
        operator bool() const noexcept { return hit; }
    };

    auto ray_box(BoundingBox const& box, Ray const& r) noexcept -> Result;
    auto ray_triangle(tcb::span<glm::vec3 const, 3> triangle, Ray const& r) noexcept -> Result;
    auto ray_sphere(float radius, Ray const& r) noexcept -> Result;
}