#pragma once

#include <glm/glm.hpp>
#include "ray.h"
#include "boundingBox.h"
#include "boundingSphere.h"

namespace PTS {
    namespace Intersection {
        struct Result {
            bool hit;
            float t;

            HOST DEVICE Result(bool hit) noexcept : hit{ hit }, t{ 0.0f } { }
            HOST DEVICE Result(bool hit, float t) noexcept : hit{ hit }, t{ t } { }
            HOST DEVICE operator bool() const noexcept { return hit; }
        };

        HOST DEVICE auto ray_box(BoundingBox const& box, Ray const& r) noexcept -> Result;
        HOST DEVICE auto ray_triangle(glm::vec3 const (&triangle) [3], Ray const& r) noexcept -> Result;
        HOST DEVICE auto ray_sphere(BoundingSphere const& sphere, Ray const& r) noexcept -> Result;
    }
}