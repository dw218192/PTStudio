#pragma once

#include <glm/common.hpp>
#include <glm/vec3.hpp>
#include <limits>

namespace pts::rendering {

struct AABB {
    glm::vec3 min{std::numeric_limits<float>::max()};
    glm::vec3 max{std::numeric_limits<float>::lowest()};

    /// Expand to include a point.
    void expand(const glm::vec3& p) {
        min = glm::min(min, p);
        max = glm::max(max, p);
    }

    /// Expand to include another AABB.
    void merge(const AABB& other) {
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
    }

    /// Half-surface area (proportional to SA, avoids the 2x multiply).
    float surface_area() const {
        auto d = max - min;
        return d.x * d.y + d.y * d.z + d.z * d.x;
    }

    /// True if no points have been added (min > max on any axis).
    bool empty() const {
        return min.x > max.x;
    }

    glm::vec3 center() const {
        return (min + max) * 0.5f;
    }

    glm::vec3 extent() const {
        return max - min;
    }

    int longest_axis() const {
        auto d = extent();
        if (d.y > d.x && d.y > d.z) return 1;
        if (d.z > d.x && d.z > d.y) return 2;
        return 0;
    }

    /// Construct from a single point.
    static AABB from_point(const glm::vec3& p) {
        return {p, p};
    }

    /// Construct from explicit min/max.
    static AABB from_min_max(const glm::vec3& lo, const glm::vec3& hi) {
        return {lo, hi};
    }
};

}  // namespace pts::rendering
