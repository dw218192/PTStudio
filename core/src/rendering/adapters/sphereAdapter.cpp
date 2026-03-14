#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/sphereAdapter.h>
#include <pxr/imaging/geomUtil/sphereMeshGenerator.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/usd/usdGeom/sphere.h>

#include <algorithm>
#include <cmath>

namespace pts::rendering {

namespace {
constexpr size_t k_num_radial = 32;
constexpr size_t k_num_axial = 16;
constexpr float k_pi = 3.14159265358979323846f;
}  // namespace

const SphereAdapter& SphereAdapter::instance() {
    static const SphereAdapter s_instance;
    return s_instance;
}

bool SphereAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomSphere>();
}

std::optional<AdapterResult> SphereAdapter::adapt(const pxr::UsdPrim& prim) const {
    pxr::UsdGeomSphere sphere(prim);

    double radius = 1.0;
    sphere.GetRadiusAttr().Get(&radius);
    float r = static_cast<float>(radius);

    auto colors = read_display_color(prim);

    auto topo = pxr::GeomUtilSphereMeshGenerator::GenerateTopology(k_num_radial, k_num_axial);
    const auto& face_counts = topo.GetFaceVertexCounts();
    const auto& face_indices = topo.GetFaceVertexIndices();

    size_t num_pts = pxr::GeomUtilSphereMeshGenerator::ComputeNumPoints(k_num_radial, k_num_axial);
    pxr::VtVec3fArray points(num_pts);
    pxr::GeomUtilSphereMeshGenerator::GeneratePoints(points.begin(), k_num_radial, k_num_axial, r);

    pxr::VtVec3fArray normals(num_pts);
    pxr::GeomUtilSphereMeshGenerator::GenerateNormals(normals.begin(), k_num_radial, k_num_axial);

    // GeomUtil sphere: cross-sections in XY, pole along Z.
    // Remap to Y-up: (x, y, z) -> (x, z, -y)
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
        float v = std::acos(std::clamp(points[i][2] / r, -1.0f, 1.0f)) / k_pi;
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
