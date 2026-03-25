#pragma once

#include <boost/core/span.hpp>
#include <cstdint>
#include <glm/vec3.hpp>
#include <vector>

namespace pts::rendering {

struct BVHNode {
    glm::vec3 aabb_min;
    uint32_t left_first;
    glm::vec3 aabb_max;
    uint32_t count;
};
static_assert(sizeof(BVHNode) == 32);

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<uint32_t> tri_indices;
    glm::vec3 scene_aabb_min{0};
    glm::vec3 scene_aabb_max{0};
};

inline constexpr uint32_t k_bvh_max_leaf_size = 4;
inline constexpr uint32_t k_bvh_bin_count = 16;

/// Build a BVH from per-triangle AABBs and centroids.
BVH build_bvh(boost::span<const glm::vec3> centroids, boost::span<const glm::vec3> aabb_mins,
              boost::span<const glm::vec3> aabb_maxs, uint32_t count);

}  // namespace pts::rendering
