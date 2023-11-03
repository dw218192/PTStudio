#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "vertex.h"
#include "utils.h"
#include "reflection.h"

struct BoundingBox {
	BEGIN_REFLECT(BoundingBox);
	    FIELD_MOD(glm::vec3, min_pos, MSerialize{});
	    FIELD_MOD(glm::vec3, max_pos, MSerialize{});
	END_REFLECT();

    static auto from_vertices(std::vector<Vertex> const& vertices) noexcept -> BoundingBox;
    NODISCARD glm::vec3 get_center() const noexcept {
        return (min_pos + max_pos) * 0.5f;
    }
    NODISCARD glm::vec3 get_extent() const noexcept {
        return (max_pos - min_pos) * 0.5f;
    }
    auto operator+=(BoundingBox const& rhs) noexcept -> BoundingBox& {
        min_pos = glm::min(min_pos, rhs.min_pos);
        max_pos = glm::max(max_pos, rhs.max_pos);
        return *this;
    }
    auto operator+(BoundingBox const& rhs) const noexcept -> BoundingBox {
        return BoundingBox{ glm::min(min_pos, rhs.min_pos), glm::max(max_pos, rhs.max_pos) };
    }
};