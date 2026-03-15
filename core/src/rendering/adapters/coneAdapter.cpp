#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/coneAdapter.h>
#include <core/rendering/renderWorld.h>
#include <pxr/imaging/geomUtil/coneMeshGenerator.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/tokens.h>

#include <cmath>

namespace pts::rendering {

namespace {
constexpr size_t k_num_radial = 32;
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

ConeAdapter& ConeAdapter::instance() {
    static ConeAdapter s_instance;
    return s_instance;
}

bool ConeAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomCone>();
}

void ConeAdapter::sync(pxr::UsdPrim prim, RenderWorld& world, const webgpu::Device& device) {
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
        float v = (height > 0.0f) ? (points[i][2] + half_h) / height : 0.0f;
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

    auto prim_path = prim.GetPath().GetString();
    auto transform = compute_world_transform(prim);
    auto material_index = resolve_material(prim, world);

    int existing = world.find_object_by_prim(prim_path);
    if (existing >= 0) {
        auto& obj = world.objects[existing];
        obj.transform = transform;
        obj.material_index = material_index;
        upload_mesh(world, device, vertices, indices, obj.mesh_index);
    } else {
        auto mesh_slot = world.alloc_mesh_slot();
        auto obj_slot = world.alloc_object_slot();
        upload_mesh(world, device, vertices, indices, mesh_slot);
        auto& obj = world.objects[obj_slot];
        obj.mesh_index = mesh_slot;
        obj.transform = transform;
        obj.material_index = material_index;
        obj.prim_path = prim_path;
        world.prim_to_object[prim_path] = obj_slot;
    }
    ++world.mesh_version;
}

}  // namespace pts::rendering
