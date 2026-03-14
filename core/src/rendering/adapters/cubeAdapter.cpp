#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/cubeAdapter.h>
#include <pxr/imaging/geomUtil/cuboidMeshGenerator.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/usd/usdGeom/cube.h>

namespace pts::rendering {

const CubeAdapter& CubeAdapter::instance() {
    static const CubeAdapter s_instance;
    return s_instance;
}

bool CubeAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomCube>();
}

std::optional<AdapterResult> CubeAdapter::adapt(const pxr::UsdPrim& prim) const {
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
    std::vector<uint32_t> indices;

    int idx_offset = 0;
    for (size_t f = 0; f < face_counts.size(); ++f) {
        int fvc = face_counts[f];
        auto base = static_cast<uint32_t>(vertices.size());
        const auto& n = normals[f];

        for (int j = 0; j < fvc; ++j) {
            int pi = face_indices[idx_offset + j];
            const auto& p = points[pi];

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

        for (int t = 1; t < fvc - 1; ++t) {
            indices.push_back(base);
            indices.push_back(base + t);
            indices.push_back(base + t + 1);
        }

        idx_offset += fvc;
    }

    return MeshResult{std::move(vertices), std::move(indices)};
}

}  // namespace pts::rendering
