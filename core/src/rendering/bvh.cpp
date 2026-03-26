#include <core/diagnostics.h>
#include <core/rendering/bvh.h>
#include <core/rendering/packedTriangle.h>
#include <core/rendering/vertex.h>
#include <core/rendering/webgpu/device.h>
#include <imgui.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>

namespace pts::rendering {
namespace {

struct SplitResult {
    float cost = std::numeric_limits<float>::max();
    int axis = -1;
    int split_bin = -1;
};

SplitResult evaluate_sah(boost::span<const uint32_t> tri_indices, boost::span<const AABB> tri_aabbs,
                         const AABB& centroid_bounds, uint32_t first, uint32_t count) {
    struct Bin {
        AABB bounds;
        uint32_t count = 0;
    };

    SplitResult best;

    for (int axis = 0; axis < 3; ++axis) {
        float extent = centroid_bounds.max[axis] - centroid_bounds.min[axis];
        if (extent <= 0.0f) continue;

        Bin bins[k_bvh_bin_count] = {};
        float inv_extent = 1.0f / extent;

        for (uint32_t i = first; i < first + count; ++i) {
            uint32_t ti = tri_indices[i];
            float t = (tri_aabbs[ti].center()[axis] - centroid_bounds.min[axis]) * inv_extent;
            int bin_idx = std::clamp(static_cast<int>(t * k_bvh_bin_count), 0,
                                     static_cast<int>(k_bvh_bin_count) - 1);
            bins[bin_idx].bounds.merge(tri_aabbs[ti]);
            bins[bin_idx].count++;
        }

        // Left sweep
        float left_area[k_bvh_bin_count - 1];
        uint32_t left_count[k_bvh_bin_count - 1];
        {
            AABB running;
            uint32_t rc = 0;
            for (uint32_t i = 0; i < k_bvh_bin_count - 1; ++i) {
                running.merge(bins[i].bounds);
                rc += bins[i].count;
                left_area[i] = rc > 0 ? running.surface_area() : 0.0f;
                left_count[i] = rc;
            }
        }

        // Right sweep
        float right_area[k_bvh_bin_count - 1];
        uint32_t right_count[k_bvh_bin_count - 1];
        {
            AABB running;
            uint32_t rc = 0;
            for (int i = static_cast<int>(k_bvh_bin_count) - 1; i > 0; --i) {
                running.merge(bins[i].bounds);
                rc += bins[i].count;
                right_area[i - 1] = rc > 0 ? running.surface_area() : 0.0f;
                right_count[i - 1] = rc;
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
                        boost::span<const AABB> tri_aabbs, uint32_t first, uint32_t count) {
    AABB bounds;
    for (uint32_t i = first; i < first + count; ++i) {
        bounds.merge(tri_aabbs[tri_indices[i]]);
    }
    node.aabb_min = bounds.min;
    node.aabb_max = bounds.max;
}

void subdivide(std::vector<BVHNode>& nodes, std::vector<uint32_t>& tri_indices,
               boost::span<const AABB> tri_aabbs, uint32_t node_idx) {
    BVHNode& node = nodes[node_idx];
    uint32_t first = node.left_first;
    uint32_t count = node.count;

    if (count <= k_bvh_max_leaf_size) return;

    // Centroid bounds for binning
    AABB centroid_bounds;
    for (uint32_t i = first; i < first + count; ++i) {
        centroid_bounds.expand(tri_aabbs[tri_indices[i]].center());
    }

    auto best = evaluate_sah(tri_indices, tri_aabbs, centroid_bounds, first, count);

    // No valid split (degenerate centroid range) — try median split
    if (best.axis < 0) {
        if (count <= 2 * k_bvh_max_leaf_size) return;

        int axis = centroid_bounds.longest_axis();
        uint32_t mid = first + count / 2;
        std::nth_element(tri_indices.begin() + first, tri_indices.begin() + mid,
                         tri_indices.begin() + first + count, [&](uint32_t a, uint32_t b) {
                             return tri_aabbs[a].center()[axis] < tri_aabbs[b].center()[axis];
                         });

        uint32_t left_count = mid - first;
        uint32_t right_count = count - left_count;

        uint32_t left_idx = static_cast<uint32_t>(nodes.size());
        nodes.push_back({});
        nodes.push_back({});

        nodes[node_idx].left_first = left_idx;
        nodes[node_idx].count = 0;

        nodes[left_idx].left_first = first;
        nodes[left_idx].count = left_count;
        update_node_bounds(nodes[left_idx], tri_indices, tri_aabbs, first, left_count);

        nodes[left_idx + 1].left_first = mid;
        nodes[left_idx + 1].count = right_count;
        update_node_bounds(nodes[left_idx + 1], tri_indices, tri_aabbs, mid, right_count);

        subdivide(nodes, tri_indices, tri_aabbs, left_idx);
        subdivide(nodes, tri_indices, tri_aabbs, left_idx + 1);
        return;
    }

    // Check if splitting is worthwhile
    AABB node_bounds = AABB::from_min_max(nodes[node_idx].aabb_min, nodes[node_idx].aabb_max);
    float no_split_cost = count * node_bounds.surface_area();
    if (best.cost >= no_split_cost) return;

    // Partition triangles by the best split
    float extent = centroid_bounds.max[best.axis] - centroid_bounds.min[best.axis];
    INVARIANT(extent > 0.0f);
    float inv_extent = 1.0f / extent;

    auto partition_point = std::partition(
        tri_indices.begin() + first, tri_indices.begin() + first + count, [&](uint32_t ti) {
            float t =
                (tri_aabbs[ti].center()[best.axis] - centroid_bounds.min[best.axis]) * inv_extent;
            int bin_idx = std::clamp(static_cast<int>(t * k_bvh_bin_count), 0,
                                     static_cast<int>(k_bvh_bin_count) - 1);
            return bin_idx <= best.split_bin;
        });

    uint32_t left_count = static_cast<uint32_t>(partition_point - (tri_indices.begin() + first));
    if (left_count == 0 || left_count == count) return;

    uint32_t right_count = count - left_count;

    uint32_t left_idx = static_cast<uint32_t>(nodes.size());
    nodes.push_back({});
    nodes.push_back({});

    nodes[node_idx].left_first = left_idx;
    nodes[node_idx].count = 0;

    nodes[left_idx].left_first = first;
    nodes[left_idx].count = left_count;
    update_node_bounds(nodes[left_idx], tri_indices, tri_aabbs, first, left_count);

    nodes[left_idx + 1].left_first = first + left_count;
    nodes[left_idx + 1].count = right_count;
    update_node_bounds(nodes[left_idx + 1], tri_indices, tri_aabbs, first + left_count,
                       right_count);

    subdivide(nodes, tri_indices, tri_aabbs, left_idx);
    subdivide(nodes, tri_indices, tri_aabbs, left_idx + 1);
}

}  // namespace

void BVH::build(boost::span<const AABB> tri_aabbs, uint32_t count) {
    PRECONDITION(tri_aabbs.size() == count);

    m_nodes.clear();
    m_tri_indices.clear();

    if (count == 0) {
        m_nodes.push_back(BVHNode{{0, 0, 0}, 0, {0, 0, 0}, 0});
        m_tree_depth = 0;
        m_debug_level = 0;
        return;
    }

    m_tri_indices.resize(count);
    std::iota(m_tri_indices.begin(), m_tri_indices.end(), 0u);

    AABB root_bounds;
    for (uint32_t i = 0; i < count; ++i) {
        root_bounds.merge(tri_aabbs[i]);
    }

    m_nodes.reserve(2 * count);
    m_nodes.push_back({});
    m_nodes[0].left_first = 0;
    m_nodes[0].count = count;
    m_nodes[0].aabb_min = root_bounds.min;
    m_nodes[0].aabb_max = root_bounds.max;

    subdivide(m_nodes, m_tri_indices, tri_aabbs, 0);

    // Compute tree depth for debug slider range
    std::function<int(uint32_t)> depth = [&](uint32_t idx) -> int {
        auto& n = m_nodes[idx];
        if (n.count > 0) return 0;
        return 1 + std::max(depth(n.left_first), depth(n.left_first + 1));
    };
    m_tree_depth = depth(0);
}

std::vector<PackedTriangle> BVH::build_from_mesh(boost::span<const Vertex> vertices,
                                                 boost::span<const uint32_t> indices) {
    PRECONDITION(!vertices.empty());
    PRECONDITION(!indices.empty());
    PRECONDITION(indices.size() % 3 == 0);

    uint32_t tri_count = static_cast<uint32_t>(indices.size()) / 3;

    // Build local-space triangle AABBs
    std::vector<AABB> tri_aabbs;
    tri_aabbs.reserve(tri_count);
    for (uint32_t i = 0; i < static_cast<uint32_t>(indices.size()); i += 3) {
        AABB a;
        for (int vi = 0; vi < 3; ++vi) {
            const auto& v = vertices[indices[i + vi]];
            a.expand(glm::vec3(v.position[0], v.position[1], v.position[2]));
        }
        tri_aabbs.push_back(a);
    }

    build(tri_aabbs, tri_count);

    // Create PackedTriangle array in local space
    std::vector<PackedTriangle> tris;
    tris.reserve(tri_count);
    for (uint32_t i = 0; i < static_cast<uint32_t>(indices.size()); i += 3) {
        const auto& v0 = vertices[indices[i + 0]];
        const auto& v1 = vertices[indices[i + 1]];
        const auto& v2 = vertices[indices[i + 2]];

        PackedTriangle tri{};
        tri.v0 = glm::vec3(v0.position[0], v0.position[1], v0.position[2]);
        tri.v1 = glm::vec3(v1.position[0], v1.position[1], v1.position[2]);
        tri.v2 = glm::vec3(v2.position[0], v2.position[1], v2.position[2]);
        tri.n0 = glm::vec3(v0.normal[0], v0.normal[1], v0.normal[2]);
        tri.n1 = glm::vec3(v1.normal[0], v1.normal[1], v1.normal[2]);
        tri.n2 = glm::vec3(v2.normal[0], v2.normal[1], v2.normal[2]);
        tri.uv0 = glm::vec2(v0.uv[0], v0.uv[1]);
        tri.uv1 = glm::vec2(v1.uv[0], v1.uv[1]);
        tri.uv2 = glm::vec2(v2.uv[0], v2.uv[1]);
        tris.push_back(tri);
    }

    // Reorder by BVH tri_indices for spatial locality
    if (!m_tri_indices.empty()) {
        INVARIANT(m_tri_indices.size() == tris.size());
        std::vector<PackedTriangle> reordered(tris.size());
        for (uint32_t i = 0; i < static_cast<uint32_t>(tris.size()); ++i) {
            reordered[i] = tris[m_tri_indices[i]];
        }
        tris = std::move(reordered);
    }

    return tris;
}

std::vector<BVHNode> BVH::concatenate_nodes(boost::span<const BlasEntry> blas_list) const {
    uint32_t tlas_nc = node_count();

    uint32_t total_blas_nodes = 0;
    for (const auto& entry : blas_list) {
        PRECONDITION(entry.bvh != nullptr);
        total_blas_nodes += entry.bvh->node_count();
    }

    std::vector<BVHNode> all_nodes;
    all_nodes.reserve(tlas_nc + total_blas_nodes);

    // TLAS nodes first
    all_nodes.insert(all_nodes.end(), m_nodes.begin(), m_nodes.end());

    // BLAS nodes with interior left_first offset
    uint32_t running_blas_offset = 0;
    for (const auto& entry : blas_list) {
        uint32_t blas_base = tlas_nc + running_blas_offset;
        for (const auto& node : entry.bvh->nodes()) {
            BVHNode offset_node = node;
            if (offset_node.count == 0) {
                offset_node.left_first += blas_base;
            }
            all_nodes.push_back(offset_node);
        }
        running_blas_offset += entry.bvh->node_count();
    }

    return all_nodes;
}

void BVH::upload(const webgpu::Device& device, WGPUQueue queue) {
    auto byte_size = std::max(sizeof(BVHNode), m_nodes.size() * sizeof(BVHNode));
    if (!m_gpu_nodes.is_valid() || m_gpu_nodes.size() < byte_size) {
        m_gpu_nodes = device.create_buffer(
            byte_size,
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
    }
    if (!m_nodes.empty()) {
        wgpuQueueWriteBuffer(queue, m_gpu_nodes.handle(), 0, m_nodes.data(),
                             m_nodes.size() * sizeof(BVHNode));
    }
}

void BVH::draw_imgui() {
    if (m_nodes.empty()) return;
    ImGui::Checkbox("Show BVH", &m_debug_enabled);
    if (m_debug_enabled) {
        ImGui::SliderInt("Level", &m_debug_level, 0, m_tree_depth);
        ImGui::Text("Nodes: %u  Depth: %d", node_count(), m_tree_depth);
    }
}

void BVH::draw_overlay(const OverlayParams& p) const {
    if (!m_debug_enabled || m_nodes.empty()) return;

    auto* draw_list = ImGui::GetWindowDrawList();
    if (!draw_list) return;

    // Project a world-space point to viewport pixel coordinates.
    auto project = [&](const glm::vec3& world) -> std::pair<ImVec2, bool> {
        auto clip = p.view_proj * glm::vec4(world, 1.0f);
        if (clip.w <= 0.0f) return {{}, false};
        auto ndc = glm::vec3(clip) / clip.w;
        float sx = p.viewport_x + (ndc.x * 0.5f + 0.5f) * p.viewport_w;
        float sy = p.viewport_y + (-ndc.y * 0.5f + 0.5f) * p.viewport_h;
        return {ImVec2(sx, sy), true};
    };

    // Draw wireframe box from AABB
    auto draw_aabb = [&](const glm::vec3& lo, const glm::vec3& hi, ImU32 col) {
        glm::vec3 corners[8];
        for (int c = 0; c < 8; ++c) {
            corners[c] = {(c & 1) ? hi.x : lo.x, (c & 2) ? hi.y : lo.y, (c & 4) ? hi.z : lo.z};
        }
        // 12 edges of a box
        static constexpr int edges[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3},
                                             {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        for (auto& e : edges) {
            auto [a, a_ok] = project(corners[e[0]]);
            auto [b, b_ok] = project(corners[e[1]]);
            if (a_ok && b_ok) {
                draw_list->AddLine(a, b, col);
            }
        }
    };

    // Level-based color (hue rotates with depth)
    auto level_color = [](int level) -> ImU32 {
        float hue = static_cast<float>(level % 6) / 6.0f;
        ImVec4 c;
        ImGui::ColorConvertHSVtoRGB(hue, 0.8f, 1.0f, c.x, c.y, c.z);
        return IM_COL32(static_cast<int>(c.x * 255), static_cast<int>(c.y * 255),
                        static_cast<int>(c.z * 255), 180);
    };

    // Walk tree, draw nodes at the target level
    struct Entry {
        uint32_t idx;
        int level;
    };
    std::vector<Entry> stack = {{0, 0}};
    while (!stack.empty()) {
        auto [idx, lvl] = stack.back();
        stack.pop_back();
        auto& n = m_nodes[idx];

        if (lvl == m_debug_level || n.count > 0) {
            draw_aabb(n.aabb_min, n.aabb_max, level_color(lvl));
        } else if (n.count == 0 && lvl < m_debug_level) {
            stack.push_back({n.left_first, lvl + 1});
            stack.push_back({n.left_first + 1, lvl + 1});
        }
    }
}

}  // namespace pts::rendering
