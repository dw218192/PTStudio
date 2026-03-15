#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/cubeAdapter.h>
#include <core/rendering/renderWorld.h>
#include <pxr/imaging/geomUtil/cuboidMeshGenerator.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/tokens.h>

namespace pts::rendering {

CubeAdapter& CubeAdapter::instance() {
    static CubeAdapter s_instance;
    return s_instance;
}

bool CubeAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomCube>();
}

void CubeAdapter::sync(pxr::UsdPrim prim, SyncScope& scope, const webgpu::Device& device) {
    pxr::UsdGeomCube cube(prim);

    double size = 2.0;
    cube.GetSizeAttr().Get(&size);
    float s = static_cast<float>(size);

    auto colors = read_display_color(prim);

    auto topo = pxr::GeomUtilCuboidMeshGenerator::GenerateTopology();
    const auto& face_counts = topo.GetFaceVertexCounts();
    const auto& face_indices = topo.GetFaceVertexIndices();

    size_t num_pts = pxr::GeomUtilCuboidMeshGenerator::ComputeNumPoints();
    pxr::VtVec3fArray points(num_pts);
    pxr::GeomUtilCuboidMeshGenerator::GeneratePoints(points.begin(), s, s, s);

    pxr::VtVec3fArray normals(6);
    pxr::GeomUtilCuboidMeshGenerator::GenerateNormals(normals.begin());

    // clang-format off
    static constexpr float k_quad_uvs[4][2] = {
        {0, 0}, {1, 0}, {1, 1}, {0, 1}
    };
    // clang-format on

    std::vector<Vertex> vertices;
    pxr::VtIntArray seq_fv_indices(face_indices.size());

    int idx_offset = 0;
    for (size_t f = 0; f < face_counts.size(); ++f) {
        int fvc = face_counts[f];
        const auto& n = normals[f];

        for (int j = 0; j < fvc; ++j) {
            int pi = face_indices[idx_offset + j];
            const auto& p = points[pi];

            seq_fv_indices[idx_offset + j] = static_cast<int>(vertices.size());

            Vertex vtx = {};
            vtx.position[0] = p[0];
            vtx.position[1] = p[1];
            vtx.position[2] = p[2];
            vtx.normal[0] = n[0];
            vtx.normal[1] = n[1];
            vtx.normal[2] = n[2];
            vtx.uv[0] = k_quad_uvs[j % 4][0];
            vtx.uv[1] = k_quad_uvs[j % 4][1];
            apply_display_color(vtx, colors);
            vertices.push_back(vtx);
        }

        idx_offset += fvc;
    }

    pxr::HdMeshTopology hd_topo(pxr::PxOsdOpenSubdivTokens->none, pxr::UsdGeomTokens->rightHanded,
                                face_counts, seq_fv_indices);
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
