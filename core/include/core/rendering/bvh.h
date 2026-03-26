#pragma once

#include <core/rendering/aabb.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/webgpu.h>

#include <boost/core/span.hpp>
#include <cstdint>
#include <glm/mat4x4.hpp>
#include <vector>

namespace pts::webgpu {
class Device;
}

namespace pts::rendering {

/// GPU-friendly BVH node (32 bytes = 2x vec4).
/// Interior: count == 0, left_first = left child index (right = left + 1).
/// Leaf: count > 0, left_first = first triangle index in reordered array.
struct BVHNode {
    glm::vec3 aabb_min;
    uint32_t left_first;
    glm::vec3 aabb_max;
    uint32_t count;
};
static_assert(sizeof(BVHNode) == 32);

inline constexpr uint32_t k_bvh_max_leaf_size = 4;
inline constexpr uint32_t k_bvh_bin_count = 16;

/// CPU-side BVH with optional GPU buffer upload.
class BVH {
   public:
    /// Build from per-triangle AABBs and centroids (binned SAH, top-down).
    void build(boost::span<const AABB> tri_aabbs, uint32_t count);

    /// Upload node array to a GPU storage buffer.
    void upload(const webgpu::Device& device, WGPUQueue queue);

    [[nodiscard]] boost::span<const BVHNode> nodes() const {
        return m_nodes;
    }
    [[nodiscard]] boost::span<const uint32_t> tri_indices() const {
        return m_tri_indices;
    }
    [[nodiscard]] uint32_t node_count() const {
        return static_cast<uint32_t>(m_nodes.size());
    }
    [[nodiscard]] const webgpu::Buffer& gpu_nodes() const {
        return m_gpu_nodes;
    }

    /// Scene AABB — just the root node's bounds.
    [[nodiscard]] AABB scene_bounds() const {
        if (m_nodes.empty()) return {};
        return AABB::from_min_max(m_nodes[0].aabb_min, m_nodes[0].aabb_max);
    }

    /// ImGui controls for BVH debug visualization.
    void draw_imgui();

    /// Draw wireframe AABBs on the current ImGui window's draw list.
    /// Call between BeginChild/EndChild of the viewport, or after setting
    /// the draw list clip rect to the viewport bounds.
    struct OverlayParams {
        glm::mat4 view_proj;
        float viewport_x, viewport_y;
        float viewport_w, viewport_h;
    };
    void draw_overlay(const OverlayParams& params) const;

   private:
    std::vector<BVHNode> m_nodes;
    std::vector<uint32_t> m_tri_indices;
    webgpu::Buffer m_gpu_nodes;

    // Debug overlay state
    bool m_debug_enabled = false;
    int m_debug_level = 0;
    int m_tree_depth = 0;
};

}  // namespace pts::rendering
