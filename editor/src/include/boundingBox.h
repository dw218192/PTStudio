#pragma once

#include "ext.h"
#include "vertex.h"

struct BoundingBox {
    glm::vec3 min_pos, max_pos;

    static auto from_vertices(std::vector<Vertex> const& vertices) noexcept -> BoundingBox;

    [[nodiscard]] glm::vec3 get_center() const noexcept {
        return (min_pos + max_pos) * 0.5f;
    }
    [[nodiscard]] glm::vec3 get_extent() const noexcept {
        return (max_pos - min_pos) * 0.5f;
    }

    BoundingBox& operator+=(BoundingBox const& rhs) noexcept {
        min_pos = glm::min(min_pos, rhs.min_pos);
        max_pos = glm::max(max_pos, rhs.max_pos);
        return *this;
    }

    BoundingBox operator+(BoundingBox const& rhs) const noexcept {
        return BoundingBox{ glm::min(min_pos, rhs.min_pos), glm::max(max_pos, rhs.max_pos) };
    }
};