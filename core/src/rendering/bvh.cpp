#include <core/diagnostics.h>
#include <core/rendering/bvh.h>

#include <algorithm>
#include <glm/common.hpp>
#include <limits>
#include <numeric>

namespace pts::rendering {
namespace {

float compute_surface_area(const glm::vec3& aabb_min, const glm::vec3& aabb_max) {
    auto d = aabb_max - aabb_min;
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

struct Bin {
    glm::vec3 aabb_min{std::numeric_limits<float>::max()};
    glm::vec3 aabb_max{std::numeric_limits<float>::lowest()};
    uint32_t count = 0;
};

struct SplitResult {
    float cost = std::numeric_limits<float>::max();
    int axis = -1;
    int split_bin = -1;
};

SplitResult evaluate_sah(boost::span<const uint32_t> tri_indices,
                         boost::span<const glm::vec3> centroids,
                         boost::span<const glm::vec3> aabb_mins,
                         boost::span<const glm::vec3> aabb_maxs, const glm::vec3& node_min,
                         const glm::vec3& node_max, uint32_t first, uint32_t count) {
    SplitResult best;

    for (int axis = 0; axis < 3; ++axis) {
        float extent = node_max[axis] - node_min[axis];
        if (extent <= 0.0f) continue;

        Bin bins[k_bvh_bin_count] = {};

        float inv_extent = 1.0f / extent;
        for (uint32_t i = first; i < first + count; ++i) {
            uint32_t ti = tri_indices[i];
            float t = (centroids[ti][axis] - node_min[axis]) * inv_extent;
            int bin_idx = static_cast<int>(t * k_bvh_bin_count);
            bin_idx = std::clamp(bin_idx, 0, static_cast<int>(k_bvh_bin_count) - 1);
            bins[bin_idx].aabb_min = glm::min(bins[bin_idx].aabb_min, aabb_mins[ti]);
            bins[bin_idx].aabb_max = glm::max(bins[bin_idx].aabb_max, aabb_maxs[ti]);
            bins[bin_idx].count++;
        }

        // Sweep from left to compute prefix counts and AABBs
        float left_area[k_bvh_bin_count - 1];
        uint32_t left_count[k_bvh_bin_count - 1];
        {
            glm::vec3 running_min{std::numeric_limits<float>::max()};
            glm::vec3 running_max{std::numeric_limits<float>::lowest()};
            uint32_t running_count = 0;
            for (uint32_t i = 0; i < k_bvh_bin_count - 1; ++i) {
                running_min = glm::min(running_min, bins[i].aabb_min);
                running_max = glm::max(running_max, bins[i].aabb_max);
                running_count += bins[i].count;
                left_area[i] =
                    (running_count > 0) ? compute_surface_area(running_min, running_max) : 0.0f;
                left_count[i] = running_count;
            }
        }

        // Sweep from right
        float right_area[k_bvh_bin_count - 1];
        uint32_t right_count[k_bvh_bin_count - 1];
        {
            glm::vec3 running_min{std::numeric_limits<float>::max()};
            glm::vec3 running_max{std::numeric_limits<float>::lowest()};
            uint32_t running_count = 0;
            for (int i = static_cast<int>(k_bvh_bin_count) - 1; i > 0; --i) {
                running_min = glm::min(running_min, bins[i].aabb_min);
                running_max = glm::max(running_max, bins[i].aabb_max);
                running_count += bins[i].count;
                right_area[i - 1] =
                    (running_count > 0) ? compute_surface_area(running_min, running_max) : 0.0f;
                right_count[i - 1] = running_count;
            }
        }

        for (uint32_t i = 0; i < k_bvh_bin_count - 1; ++i) {
            float cost = left_count[i] * left_area[i] + right_count[i] * right_area[i];
            if (cost < best.cost) {
                best.cost = cost;
                best.axis = axis;
                best.split_bin = static_cast<int>(i);
            }
        }
    }

    return best;
}

void update_node_bounds(BVHNode& node, boost::span<const uint32_t> tri_indices,
                        boost::span<const glm::vec3> aabb_mins,
                        boost::span<const glm::vec3> aabb_maxs, uint32_t first, uint32_t count) {
    glm::vec3 node_min{std::numeric_limits<float>::max()};
    glm::vec3 node_max{std::numeric_limits<float>::lowest()};
    for (uint32_t i = first; i < first + count; ++i) {
        uint32_t ti = tri_indices[i];
        node_min = glm::min(node_min, aabb_mins[ti]);
        node_max = glm::max(node_max, aabb_maxs[ti]);
    }
    node.aabb_min = node_min;
    node.aabb_max = node_max;
}

void subdivide(std::vector<BVHNode>& nodes, std::vector<uint32_t>& tri_indices,
               boost::span<const glm::vec3> centroids, boost::span<const glm::vec3> aabb_mins,
               boost::span<const glm::vec3> aabb_maxs, uint32_t node_idx) {
    BVHNode& node = nodes[node_idx];
    uint32_t first = node.left_first;
    uint32_t count = node.count;

    if (count <= k_bvh_max_leaf_size) return;

    // Compute centroid bounds for binning
    glm::vec3 centroid_min{std::numeric_limits<float>::max()};
    glm::vec3 centroid_max{std::numeric_limits<float>::lowest()};
    for (uint32_t i = first; i < first + count; ++i) {
        uint32_t ti = tri_indices[i];
        centroid_min = glm::min(centroid_min, centroids[ti]);
        centroid_max = glm::max(centroid_max, centroids[ti]);
    }

    SplitResult best = evaluate_sah(tri_indices, centroids, aabb_mins, aabb_maxs, centroid_min,
                                    centroid_max, first, count);

    // No valid split found (degenerate centroid range) — try median split
    if (best.axis < 0) {
        if (count <= 2 * k_bvh_max_leaf_size) return;

        // Median split on the longest axis
        auto extent = centroid_max - centroid_min;
        int axis = 0;
        if (extent.y > extent[axis]) axis = 1;
        if (extent.z > extent[axis]) axis = 2;

        uint32_t mid = first + count / 2;
        std::nth_element(tri_indices.begin() + first, tri_indices.begin() + mid,
                         tri_indices.begin() + first + count, [&](uint32_t a, uint32_t b) {
                             return centroids[a][axis] < centroids[b][axis];
                         });

        uint32_t left_count = mid - first;
        uint32_t right_count = count - left_count;

        uint32_t left_idx = static_cast<uint32_t>(nodes.size());
        nodes.push_back({});
        nodes.push_back({});

        // Re-fetch node reference after push_back (may invalidate)
        nodes[node_idx].left_first = left_idx;
        nodes[node_idx].count = 0;

        nodes[left_idx].left_first = first;
        nodes[left_idx].count = left_count;
        update_node_bounds(nodes[left_idx], tri_indices, aabb_mins, aabb_maxs, first, left_count);

        nodes[left_idx + 1].left_first = mid;
        nodes[left_idx + 1].count = right_count;
        update_node_bounds(nodes[left_idx + 1], tri_indices, aabb_mins, aabb_maxs, mid,
                           right_count);

        subdivide(nodes, tri_indices, centroids, aabb_mins, aabb_maxs, left_idx);
        subdivide(nodes, tri_indices, centroids, aabb_mins, aabb_maxs, left_idx + 1);
        return;
    }

    // Check if splitting is worthwhile
    float no_split_cost =
        count * compute_surface_area(nodes[node_idx].aabb_min, nodes[node_idx].aabb_max);
    if (best.cost >= no_split_cost) return;

    // Partition triangles by the best split
    float extent = centroid_max[best.axis] - centroid_min[best.axis];
    INVARIANT(extent > 0.0f);
    float inv_extent = 1.0f / extent;

    auto partition_point = std::partition(
        tri_indices.begin() + first, tri_indices.begin() + first + count, [&](uint32_t ti) {
            float t = (centroids[ti][best.axis] - centroid_min[best.axis]) * inv_extent;
            int bin_idx = static_cast<int>(t * k_bvh_bin_count);
            bin_idx = std::clamp(bin_idx, 0, static_cast<int>(k_bvh_bin_count) - 1);
            return bin_idx <= best.split_bin;
        });

    uint32_t left_count = static_cast<uint32_t>(partition_point - (tri_indices.begin() + first));

    // Degenerate partition — all went to one side
    if (left_count == 0 || left_count == count) return;

    uint32_t right_count = count - left_count;

    uint32_t left_idx = static_cast<uint32_t>(nodes.size());
    nodes.push_back({});
    nodes.push_back({});

    // Re-fetch after push_back
    nodes[node_idx].left_first = left_idx;
    nodes[node_idx].count = 0;

    nodes[left_idx].left_first = first;
    nodes[left_idx].count = left_count;
    update_node_bounds(nodes[left_idx], tri_indices, aabb_mins, aabb_maxs, first, left_count);

    nodes[left_idx + 1].left_first = first + left_count;
    nodes[left_idx + 1].count = right_count;
    update_node_bounds(nodes[left_idx + 1], tri_indices, aabb_mins, aabb_maxs, first + left_count,
                       right_count);

    subdivide(nodes, tri_indices, centroids, aabb_mins, aabb_maxs, left_idx);
    subdivide(nodes, tri_indices, centroids, aabb_mins, aabb_maxs, left_idx + 1);
}

}  // namespace

BVH build_bvh(boost::span<const glm::vec3> centroids, boost::span<const glm::vec3> aabb_mins,
              boost::span<const glm::vec3> aabb_maxs, uint32_t count) {
    PRECONDITION(centroids.size() == count);
    PRECONDITION(aabb_mins.size() == count);
    PRECONDITION(aabb_maxs.size() == count);

    BVH bvh;

    if (count == 0) {
        bvh.nodes.push_back(BVHNode{{0, 0, 0}, 0, {0, 0, 0}, 0});
        return bvh;
    }

    // Initialize triangle index array
    bvh.tri_indices.resize(count);
    std::iota(bvh.tri_indices.begin(), bvh.tri_indices.end(), 0u);

    // Compute scene AABB
    glm::vec3 scene_min{std::numeric_limits<float>::max()};
    glm::vec3 scene_max{std::numeric_limits<float>::lowest()};
    for (uint32_t i = 0; i < count; ++i) {
        scene_min = glm::min(scene_min, aabb_mins[i]);
        scene_max = glm::max(scene_max, aabb_maxs[i]);
    }
    bvh.scene_aabb_min = scene_min;
    bvh.scene_aabb_max = scene_max;

    // Reserve conservative upper bound for nodes (2*N - 1 for a full binary tree)
    bvh.nodes.reserve(2 * count);

    // Create root node
    bvh.nodes.push_back({});
    bvh.nodes[0].left_first = 0;
    bvh.nodes[0].count = count;
    bvh.nodes[0].aabb_min = scene_min;
    bvh.nodes[0].aabb_max = scene_max;

    subdivide(bvh.nodes, bvh.tri_indices, centroids, aabb_mins, aabb_maxs, 0);

    return bvh;
}

}  // namespace pts::rendering
