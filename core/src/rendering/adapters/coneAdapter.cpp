#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/coneAdapter.h>
#include <pxr/usd/usdGeom/cone.h>

#include <cmath>

namespace pts::rendering {

namespace {
constexpr int k_radial_segments = 32;
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

const ConeAdapter& ConeAdapter::instance() {
    static const ConeAdapter s_instance;
    return s_instance;
}

bool ConeAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomCone>();
}

std::optional<AdapterResult> ConeAdapter::adapt(const pxr::UsdPrim& prim) const {
    pxr::UsdGeomCone cone(prim);

    double radius_d = 1.0;
    double height_d = 2.0;
    cone.GetRadiusAttr().Get(&radius_d);
    cone.GetHeightAttr().Get(&height_d);
    float radius = static_cast<float>(radius_d);
    float half_h = static_cast<float>(height_d) * 0.5f;

    pxr::TfToken axis;
    cone.GetAxisAttr().Get(&axis);
    auto mapping = get_axis_mapping(axis);

    auto colors = read_display_color(prim);

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // Slope normal: for a cone with radius r and height h,
    // the side normal's longitudinal component is r/slant_length,
    // and the radial component is h/slant_length.
    float slant = std::sqrt(radius * radius + (2.0f * half_h) * (2.0f * half_h));
    float n_along_side = radius / slant;
    float n_radial_side = (2.0f * half_h) / slant;

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

    // Side triangles: apex at +half_h, base ring at -half_h
    // Each segment gets its own apex vertex for correct normals
    for (int i = 0; i <= k_radial_segments; ++i) {
        float angle = static_cast<float>(i) / k_radial_segments * 2.0f * k_pi;
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);
        float u_coord = static_cast<float>(i) / k_radial_segments;

        // Base vertex
        push_vertex(-half_h, radius * cos_a, radius * sin_a, n_along_side, n_radial_side * cos_a,
                    n_radial_side * sin_a, u_coord, 0.0f);
        // Apex vertex (duplicated per segment for correct normals/UVs)
        push_vertex(half_h, 0, 0, n_along_side, n_radial_side * cos_a, n_radial_side * sin_a,
                    u_coord, 1.0f);
    }

    // Side indices
    for (int i = 0; i < k_radial_segments; ++i) {
        uint32_t b = static_cast<uint32_t>(i * 2);
        uint32_t a = b + 1;   // apex for this segment
        uint32_t bn = b + 2;  // next base
        indices.push_back(b);
        indices.push_back(bn);
        indices.push_back(a);
    }

    // Base cap
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
