#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/capsuleAdapter.h>
#include <pxr/usd/usdGeom/capsule.h>

#include <cmath>

namespace pts::rendering {

namespace {
constexpr int k_radial_segments = 32;
constexpr int k_cap_segments = 8;
constexpr float k_pi = 3.14159265358979323846f;

struct AxisMapping {
    int along;
    int u_ax;
    int v_ax;
};

inline AxisMapping get_axis_mapping(const pxr::TfToken& axis) {
    if (axis == pxr::UsdGeomTokens->x) return {0, 1, 2};
    if (axis == pxr::UsdGeomTokens->z) return {2, 0, 1};
    return {1, 0, 2};
}
}  // namespace

const CapsuleAdapter& CapsuleAdapter::instance() {
    static const CapsuleAdapter s_instance;
    return s_instance;
}

bool CapsuleAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomCapsule>();
}

std::optional<AdapterResult> CapsuleAdapter::adapt(const pxr::UsdPrim& prim) const {
    pxr::UsdGeomCapsule capsule(prim);

    double radius_d = 0.5;
    double height_d = 1.0;
    capsule.GetRadiusAttr().Get(&radius_d);
    capsule.GetHeightAttr().Get(&height_d);
    float radius = static_cast<float>(radius_d);
    float half_h = static_cast<float>(height_d) * 0.5f;

    pxr::TfToken axis;
    capsule.GetAxisAttr().Get(&axis);
    auto mapping = get_axis_mapping(axis);

    auto colors = read_display_color(prim);

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    int cols = k_radial_segments + 1;

    auto push_vertex = [&](float along_val, float u_val, float v_val, float n_along, float n_u,
                           float n_v, float uv_u, float uv_v) {
        Vertex vtx = {};
        vtx.position[mapping.along] = along_val;
        vtx.position[mapping.u_ax] = u_val;
        vtx.position[mapping.v_ax] = v_val;
        vtx.normal[mapping.along] = n_along;
        vtx.normal[mapping.u_ax] = n_u;
        vtx.normal[mapping.v_ax] = n_v;
        vtx.uv[0] = uv_u;
        vtx.uv[1] = uv_v;
        apply_display_color(vtx, colors);
        vertices.push_back(vtx);
    };

    // Total v-segments: cap_segments (top hemi) + 1 (cylinder body) + cap_segments (bottom hemi)
    int total_rows = k_cap_segments + 1 + k_cap_segments;
    float total_v = static_cast<float>(total_rows);

    // Top hemisphere (from pole at +half_h+radius down to equator at +half_h)
    for (int lat = 0; lat <= k_cap_segments; ++lat) {
        float theta = static_cast<float>(lat) / k_cap_segments * (k_pi * 0.5f);
        float sin_t = std::sin(theta);
        float cos_t = std::cos(theta);
        float along_pos = half_h + radius * cos_t;
        float v_coord = static_cast<float>(lat) / total_v;

        for (int lon = 0; lon <= k_radial_segments; ++lon) {
            float phi = static_cast<float>(lon) / k_radial_segments * 2.0f * k_pi;
            float cos_p = std::cos(phi);
            float sin_p = std::sin(phi);

            push_vertex(along_pos, radius * sin_t * cos_p, radius * sin_t * sin_p, cos_t,
                        sin_t * cos_p, sin_t * sin_p, static_cast<float>(lon) / k_radial_segments,
                        v_coord);
        }
    }

    // Cylinder body: bottom ring (at -half_h)
    {
        float v_coord = static_cast<float>(k_cap_segments + 1) / total_v;
        for (int lon = 0; lon <= k_radial_segments; ++lon) {
            float phi = static_cast<float>(lon) / k_radial_segments * 2.0f * k_pi;
            float cos_p = std::cos(phi);
            float sin_p = std::sin(phi);

            push_vertex(-half_h, radius * cos_p, radius * sin_p, 0, cos_p, sin_p,
                        static_cast<float>(lon) / k_radial_segments, v_coord);
        }
    }

    // Bottom hemisphere (from equator at -half_h down to pole at -half_h-radius)
    for (int lat = 1; lat <= k_cap_segments; ++lat) {
        float theta = k_pi * 0.5f + static_cast<float>(lat) / k_cap_segments * (k_pi * 0.5f);
        float sin_t = std::sin(theta);
        float cos_t = std::cos(theta);
        float along_pos = -half_h + radius * cos_t;
        float v_coord = static_cast<float>(k_cap_segments + 1 + lat) / total_v;

        for (int lon = 0; lon <= k_radial_segments; ++lon) {
            float phi = static_cast<float>(lon) / k_radial_segments * 2.0f * k_pi;
            float cos_p = std::cos(phi);
            float sin_p = std::sin(phi);

            push_vertex(along_pos, radius * sin_t * cos_p, radius * sin_t * sin_p, cos_t,
                        sin_t * cos_p, sin_t * sin_p, static_cast<float>(lon) / k_radial_segments,
                        v_coord);
        }
    }

    // Generate indices for all rows
    int total_ring_rows = k_cap_segments + 1 + k_cap_segments;
    for (int row = 0; row < total_ring_rows; ++row) {
        for (int lon = 0; lon < k_radial_segments; ++lon) {
            uint32_t a = static_cast<uint32_t>(row * cols + lon);
            uint32_t b = a + 1;
            uint32_t c = static_cast<uint32_t>((row + 1) * cols + lon);
            uint32_t d = c + 1;

            // Skip degenerate triangles at poles
            if (row != 0) {
                indices.push_back(a);
                indices.push_back(c);
                indices.push_back(b);
            }
            if (row != total_ring_rows - 1) {
                indices.push_back(b);
                indices.push_back(c);
                indices.push_back(d);
            }
        }
    }

    return MeshResult{std::move(vertices), std::move(indices)};
}

}  // namespace pts::rendering
