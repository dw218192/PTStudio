#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/cylinderAdapter.h>
#include <pxr/usd/usdGeom/cylinder.h>

#include <cmath>

namespace pts::rendering {

namespace {
constexpr int k_radial_segments = 32;
constexpr float k_pi = 3.14159265358979323846f;

// Maps axis-aligned coordinates to (x,y,z) based on USD axis token.
// USD cylinders default to Y-axis. Axis maps: X→(y,z,x), Y→(x,z,y), Z→(x,y,z).
struct AxisMapping {
    int along;  // index of the cylinder's longitudinal axis
    int u_ax;   // first radial axis
    int v_ax;   // second radial axis
};

inline AxisMapping get_axis_mapping(const pxr::TfToken& axis) {
    if (axis == pxr::UsdGeomTokens->x) return {0, 1, 2};
    if (axis == pxr::UsdGeomTokens->z) return {2, 0, 1};
    return {1, 0, 2};  // Y (default)
}
}  // namespace

const CylinderAdapter& CylinderAdapter::instance() {
    static const CylinderAdapter s_instance;
    return s_instance;
}

bool CylinderAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomCylinder>();
}

std::optional<AdapterResult> CylinderAdapter::adapt(const pxr::UsdPrim& prim) const {
    pxr::UsdGeomCylinder cyl(prim);

    double radius_d = 1.0;
    double height_d = 2.0;
    cyl.GetRadiusAttr().Get(&radius_d);
    cyl.GetHeightAttr().Get(&height_d);
    float radius = static_cast<float>(radius_d);
    float half_h = static_cast<float>(height_d) * 0.5f;

    pxr::TfToken axis;
    cyl.GetAxisAttr().Get(&axis);
    auto mapping = get_axis_mapping(axis);

    auto colors = read_display_color(prim);

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

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

    // Side wall: two rings (top and bottom)
    for (int i = 0; i <= k_radial_segments; ++i) {
        float angle = static_cast<float>(i) / k_radial_segments * 2.0f * k_pi;
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);
        float u_coord = static_cast<float>(i) / k_radial_segments;

        // Bottom ring
        push_vertex(-half_h, radius * cos_a, radius * sin_a, 0, cos_a, sin_a, u_coord, 0.0f);
        // Top ring
        push_vertex(half_h, radius * cos_a, radius * sin_a, 0, cos_a, sin_a, u_coord, 1.0f);
    }

    // Side indices
    for (int i = 0; i < k_radial_segments; ++i) {
        uint32_t b = static_cast<uint32_t>(i * 2);
        uint32_t t = b + 1;
        uint32_t bn = b + 2;
        uint32_t tn = b + 3;
        indices.push_back(b);
        indices.push_back(bn);
        indices.push_back(t);
        indices.push_back(t);
        indices.push_back(bn);
        indices.push_back(tn);
    }

    // Top cap
    {
        auto center_idx = static_cast<uint32_t>(vertices.size());
        push_vertex(half_h, 0, 0, 1, 0, 0, 0.5f, 0.5f);
        auto ring_start = static_cast<uint32_t>(vertices.size());
        for (int i = 0; i <= k_radial_segments; ++i) {
            float angle = static_cast<float>(i) / k_radial_segments * 2.0f * k_pi;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            push_vertex(half_h, radius * cos_a, radius * sin_a, 1, 0, 0, 0.5f + 0.5f * cos_a,
                        0.5f + 0.5f * sin_a);
        }
        for (int i = 0; i < k_radial_segments; ++i) {
            indices.push_back(center_idx);
            indices.push_back(ring_start + i);
            indices.push_back(ring_start + i + 1);
        }
    }

    // Bottom cap
    {
        auto center_idx = static_cast<uint32_t>(vertices.size());
        push_vertex(-half_h, 0, 0, -1, 0, 0, 0.5f, 0.5f);
        auto ring_start = static_cast<uint32_t>(vertices.size());
        for (int i = 0; i <= k_radial_segments; ++i) {
            float angle = static_cast<float>(i) / k_radial_segments * 2.0f * k_pi;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            push_vertex(-half_h, radius * cos_a, radius * sin_a, -1, 0, 0, 0.5f + 0.5f * cos_a,
                        0.5f + 0.5f * sin_a);
        }
        for (int i = 0; i < k_radial_segments; ++i) {
            indices.push_back(center_idx);
            indices.push_back(ring_start + i + 1);
            indices.push_back(ring_start + i);
        }
    }

    return MeshResult{std::move(vertices), std::move(indices)};
}

}  // namespace pts::rendering
