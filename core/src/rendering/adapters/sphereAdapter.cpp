#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/sphereAdapter.h>
#include <core/rendering/renderWorld.h>
#include <pxr/imaging/geomUtil/sphereMeshGenerator.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/tokens.h>

#include <algorithm>
#include <cmath>

namespace pts::rendering {

namespace {
constexpr size_t k_num_radial = 32;
constexpr size_t k_num_axial = 16;
constexpr float k_pi = 3.14159265358979323846f;
}  // namespace

SphereAdapter& SphereAdapter::instance() {
    static SphereAdapter s_instance;
    return s_instance;
}

bool SphereAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomSphere>();
}

void SphereAdapter::sync(pxr::UsdPrim prim, SyncScope& scope, const webgpu::Device& device) {
    pxr::UsdGeomSphere sphere(prim);

    double radius = 1.0;
    sphere.GetRadiusAttr().Get(&radius);
    float r = static_cast<float>(radius);

    auto colors = read_display_color(prim);

    auto topo = pxr::GeomUtilSphereMeshGenerator::GenerateTopology(k_num_radial, k_num_axial);

    size_t num_pts = pxr::GeomUtilSphereMeshGenerator::ComputeNumPoints(k_num_radial, k_num_axial);
    pxr::VtVec3fArray points(num_pts);
    pxr::GeomUtilSphereMeshGenerator::GeneratePoints(points.begin(), k_num_radial, k_num_axial, r);

    pxr::VtVec3fArray normals(num_pts);
    pxr::GeomUtilSphereMeshGenerator::GenerateNormals(normals.begin(), k_num_radial, k_num_axial);

    std::vector<Vertex> vertices(num_pts);
    for (size_t i = 0; i < num_pts; ++i) {
        Vertex& vtx = vertices[i];
        vtx.position[0] = points[i][0];
        vtx.position[1] = points[i][2];
        vtx.position[2] = -points[i][1];
        vtx.normal[0] = normals[i][0];
        vtx.normal[1] = normals[i][2];
        vtx.normal[2] = -normals[i][1];

        float u = std::atan2(points[i][1], points[i][0]) / (2.0f * k_pi) + 0.5f;
        float v = (r > 0.0f) ? std::acos(std::clamp(points[i][2] / r, -1.0f, 1.0f)) / k_pi : 0.0f;
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
        indices.push_back(static_cast<uint32_t>(tri[1]));
        indices.push_back(static_cast<uint32_t>(tri[2]));
    }

    sync_object(prim, scope, device, vertices, indices);
}

}  // namespace pts::rendering
