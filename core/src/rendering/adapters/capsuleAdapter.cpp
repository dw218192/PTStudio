#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/capsuleAdapter.h>
#include <core/rendering/renderWorld.h>
#include <pxr/imaging/geomUtil/capsuleMeshGenerator.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/tokens.h>

#include <cmath>

namespace pts::rendering {

namespace {
constexpr size_t k_num_radial = 32;
constexpr size_t k_num_cap_axial = 8;
constexpr float k_pi = 3.14159265358979323846f;

struct AxisMapping {
    int along;
    int u_ax;
    int v_ax;
    bool flip_winding;
};

inline AxisMapping get_axis_mapping(const pxr::TfToken& axis) {
    if (axis == pxr::UsdGeomTokens->x) return {0, 1, 2, false};
    if (axis == pxr::UsdGeomTokens->z) return {2, 0, 1, false};
    return {1, 0, 2, true};
}
}  // namespace

CapsuleAdapter& CapsuleAdapter::instance() {
    static CapsuleAdapter s_instance;
    return s_instance;
}

bool CapsuleAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomCapsule>();
}

void CapsuleAdapter::sync(pxr::UsdPrim prim, SyncScope& scope, const webgpu::Device& device) {
    pxr::UsdGeomCapsule capsule(prim);

    double radius_d = 0.5;
    double height_d = 1.0;
    capsule.GetRadiusAttr().Get(&radius_d);
    capsule.GetHeightAttr().Get(&height_d);
    float radius = static_cast<float>(radius_d);
    float height = static_cast<float>(height_d);

    pxr::TfToken axis;
    capsule.GetAxisAttr().Get(&axis);
    auto mapping = get_axis_mapping(axis);

    auto colors = read_display_color(prim);

    auto topo = pxr::GeomUtilCapsuleMeshGenerator::GenerateTopology(k_num_radial, k_num_cap_axial);

    size_t num_pts =
        pxr::GeomUtilCapsuleMeshGenerator::ComputeNumPoints(k_num_radial, k_num_cap_axial);
    pxr::VtVec3fArray points(num_pts);
    pxr::GeomUtilCapsuleMeshGenerator::GeneratePoints(points.begin(), k_num_radial, k_num_cap_axial,
                                                      radius, height);

    pxr::VtVec3fArray normals(num_pts);
    pxr::GeomUtilCapsuleMeshGenerator::GenerateNormals(normals.begin(), k_num_radial,
                                                       k_num_cap_axial, radius, height);

    float half_h = height * 0.5f;
    float total_extent = height + 2.0f * radius;

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
        float v = (total_extent > 0.0f) ? (points[i][2] + half_h + radius) / total_extent : 0.0f;
        vtx.uv[0] = u;
        vtx.uv[1] = v;
        apply_display_color(vtx, colors);
    }

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

    sync_object(prim, scope, device, vertices, indices);
}

}  // namespace pts::rendering
