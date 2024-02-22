#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <limits>
#include "vertex.h"
#include "utils.h"

#include "object/reflection.h"

namespace PTS {
    struct BoundingBox {
        BoundingBox() noexcept : min_pos{ std::numeric_limits<float>::max() }, max_pos{ std::numeric_limits<float>::lowest() } { }
        BoundingBox(glm::vec3 min_pos, glm::vec3 max_pos) noexcept : min_pos{ min_pos }, max_pos{ max_pos } { }

        BEGIN_REFLECT(BoundingBox, void);
        FIELD(glm::vec3, min_pos, {},
            MSerialize{});
        FIELD(glm::vec3, max_pos, {},
            MSerialize{});
        END_REFLECT();

    public:
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
}