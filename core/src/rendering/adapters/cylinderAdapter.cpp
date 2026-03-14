#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/cylinderAdapter.h>
#include <pxr/imaging/geomUtil/cylinderMeshGenerator.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/tokens.h>

#include <cmath>

namespace pts::rendering {

namespace {
constexpr size_t k_num_radial = 32;
constexpr float k_pi = 3.14159265358979323846f;

// GeomUtil generates along Z. This maps Z-aligned geometry to the requested axis.
struct AxisMapping {
    int along;          // index of the cylinder's longitudinal axis
    int u_ax;           // first radial axis
    int v_ax;           // second radial axis
    bool flip_winding;  // true when the permutation is odd (flips handedness)
};

inline AxisMapping get_axis_mapping(const pxr::TfToken& axis) {
    if (axis == pxr::UsdGeomTokens->x) return {0, 1, 2, false};
    if (axis == pxr::UsdGeomTokens->z) return {2, 0, 1, false};
    return {1, 0, 2, true};  // Y — swapping Y↔Z is an odd permutation
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
    float height = static_cast<float>(height_d);

    pxr::TfToken axis;
    cyl.GetAxisAttr().Get(&axis);
    auto mapping = get_axis_mapping(axis);

    auto colors = read_display_color(prim);

    auto topo = pxr::GeomUtilCylinderMeshGenerator::GenerateTopology(k_num_radial);

    size_t num_pts = pxr::GeomUtilCylinderMeshGenerator::ComputeNumPoints(k_num_radial);
    pxr::VtVec3fArray points(num_pts);
    pxr::GeomUtilCylinderMeshGenerator::GeneratePoints(points.begin(), k_num_radial, radius,
                                                       height);

    pxr::VtVec3fArray normals(num_pts);
    pxr::GeomUtilCylinderMeshGenerator::GenerateNormals(normals.begin(), k_num_radial, radius,
                                                        height);

    float half_h = height * 0.5f;

    // Build vertices: remap from Z-aligned to requested axis
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

    // Triangulate via HdMeshUtil
    pxr::HdMeshTopology hd_topo(pxr::PxOsdOpenSubdivTokens->none, pxr::UsdGeomTokens->rightHanded,
                                topo.GetFaceVertexCounts(), topo.GetFaceVertexIndices());
    pxr::HdMeshUtil mesh_util(&hd_topo, pxr::SdfPath());
    pxr::VtVec3iArray tri_indices;
    pxr::VtIntArray prim_params;
    mesh_util.ComputeTriangleIndices(&tri_indices, &prim_params);

    std::vector<uint32_t> indices;
    indices.reserve(tri_indices.size() * 3);
    for (const auto& tri : tri_indices) {
        indices.push_back(static_cast<uint32_t>(tri[0]));
        if (mapping.flip_winding) {
            indices.push_back(static_cast<uint32_t>(tri[2]));
            indices.push_back(static_cast<uint32_t>(tri[1]));
        } else {
            indices.push_back(static_cast<uint32_t>(tri[1]));
            indices.push_back(static_cast<uint32_t>(tri[2]));
        }
    }

    return MeshResult{std::move(vertices), std::move(indices)};
}

}  // namespace pts::rendering
