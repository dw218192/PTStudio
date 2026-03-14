#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/coneAdapter.h>
#include <pxr/imaging/geomUtil/coneMeshGenerator.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/usd/usdGeom/cone.h>

#include <cmath>

namespace pts::rendering {

namespace {
constexpr size_t k_num_radial = 32;
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
    float height = static_cast<float>(height_d);

    pxr::TfToken axis;
    cone.GetAxisAttr().Get(&axis);
    auto mapping = get_axis_mapping(axis);

    auto colors = read_display_color(prim);

    auto topo = pxr::GeomUtilConeMeshGenerator::GenerateTopology(k_num_radial);
    const auto& face_counts = topo.GetFaceVertexCounts();
    const auto& face_indices = topo.GetFaceVertexIndices();

    size_t num_pts = pxr::GeomUtilConeMeshGenerator::ComputeNumPoints(k_num_radial);
    pxr::VtVec3fArray points(num_pts);
    pxr::GeomUtilConeMeshGenerator::GeneratePoints(points.begin(), k_num_radial, radius, height);

    pxr::VtVec3fArray normals(num_pts);
    pxr::GeomUtilConeMeshGenerator::GenerateNormals(normals.begin(), k_num_radial, radius, height);

    float half_h = height * 0.5f;

    std::vector<Vertex> vertices(num_pts);
    for (size_t i = 0; i < num_pts; ++i) {
        Vertex& vtx = vertices[i];
        vtx.position[mapping.along] = points[i][2];
        vtx.position[mapping.u_ax] = points[i][0];
        vtx.position[mapping.v_ax] = points[i][1];
        vtx.normal[mapping.along] = normals[i][2];
        vtx.normal[mapping.u_ax] = normals[i][0];
        vtx.normal[mapping.v_ax] = normals[i][1];

        float u = std::atan2(points[i][1], points[i][0]) / (2.0f * k_pi) + 0.5f;
        float v = (points[i][2] + half_h) / height;
        vtx.uv[0] = u;
        vtx.uv[1] = v;
        apply_display_color(vtx, colors);
    }

    // Triangulate
    std::vector<uint32_t> indices;
    int idx_offset = 0;
    for (size_t f = 0; f < face_counts.size(); ++f) {
        int fvc = face_counts[f];
        auto v0 = static_cast<uint32_t>(face_indices[idx_offset]);
        for (int t = 1; t < fvc - 1; ++t) {
            auto v1 = static_cast<uint32_t>(face_indices[idx_offset + t]);
            auto v2 = static_cast<uint32_t>(face_indices[idx_offset + t + 1]);
            indices.push_back(v0);
            indices.push_back(v1);
            indices.push_back(v2);
        }
        idx_offset += fvc;
    }

    return MeshResult{std::move(vertices), std::move(indices)};
}

}  // namespace pts::rendering
