#pragma once

#include <variant>

#include "boundingBox.h"
#include "boundingSphere.h"
#include "intersection.h"
#include "utils.h"
namespace PTS {
using BoundingVolume = std::variant<BoundingBox, BoundingSphere>;

struct RayIntersectionVisiter {
    NO_COPY_MOVE(RayIntersectionVisiter);

    RayIntersectionVisiter(Ray const& ray) noexcept : m_ray{ray} {
    }
    ~RayIntersectionVisiter() noexcept = default;

    auto operator()(BoundingBox const& box) const noexcept -> Intersection::Result {
        return Intersection::ray_box(box, m_ray);
    }
    auto operator()(BoundingSphere const& sphere) const noexcept -> Intersection::Result {
        return Intersection::ray_sphere(sphere, m_ray);
    }

   private:
    Ray const& m_ray;
};
}  // namespace PTS