#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/meshAdapter.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>

namespace pts::rendering {

const MeshAdapter& MeshAdapter::instance() {
    static const MeshAdapter s_instance;
    return s_instance;
}

bool MeshAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomMesh>();
}

std::optional<AdapterResult> MeshAdapter::adapt(const pxr::UsdPrim& prim) const {
    pxr::UsdGeomMesh mesh(prim);

    pxr::VtVec3fArray points;
    mesh.GetPointsAttr().Get(&points);
    if (points.empty()) return std::nullopt;

    pxr::VtVec3fArray normals;
    mesh.GetNormalsAttr().Get(&normals);

    pxr::VtIntArray face_vertex_counts;
    mesh.GetFaceVertexCountsAttr().Get(&face_vertex_counts);

    pxr::VtIntArray face_vertex_indices;
    mesh.GetFaceVertexIndicesAttr().Get(&face_vertex_indices);

    auto primvars_api = pxr::UsdGeomPrimvarsAPI(prim);

    auto display_colors = read_display_color(prim);

    pxr::VtVec2fArray uvs;
    auto uv_pv = primvars_api.GetPrimvar(pxr::TfToken("st"));
    if (uv_pv) {
        uv_pv.Get(&uvs);
    }

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    size_t fv_offset = 0;
    for (size_t face = 0; face < face_vertex_counts.size(); face++) {
        int count = face_vertex_counts[face];
        if (count < 3) {
            fv_offset += count;
            continue;
        }

        pxr::GfVec3f face_normal(0, 0, 0);
        if (normals.empty()) {
            int i0 = face_vertex_indices[fv_offset];
            int i1 = face_vertex_indices[fv_offset + 1];
            int i2 = face_vertex_indices[fv_offset + 2];
            pxr::GfVec3f e1 = points[i1] - points[i0];
            pxr::GfVec3f e2 = points[i2] - points[i0];
            face_normal = e1 ^ e2;
            float len = face_normal.GetLength();
            if (len > 0) face_normal /= len;
        }

        uint32_t base = static_cast<uint32_t>(vertices.size());

        for (int j = 0; j < count; j++) {
            int pt_idx = face_vertex_indices[fv_offset + j];
            Vertex v = {};

            v.position[0] = points[pt_idx][0];
            v.position[1] = points[pt_idx][1];
            v.position[2] = points[pt_idx][2];

            if (!normals.empty()) {
                if (normals.size() == face_vertex_indices.size()) {
                    v.normal[0] = normals[fv_offset + j][0];
                    v.normal[1] = normals[fv_offset + j][1];
                    v.normal[2] = normals[fv_offset + j][2];
                } else if (normals.size() == points.size()) {
                    v.normal[0] = normals[pt_idx][0];
                    v.normal[1] = normals[pt_idx][1];
                    v.normal[2] = normals[pt_idx][2];
                } else {
                    v.normal[0] = face_normal[0];
                    v.normal[1] = face_normal[1];
                    v.normal[2] = face_normal[2];
                }
            } else {
                v.normal[0] = face_normal[0];
                v.normal[1] = face_normal[1];
                v.normal[2] = face_normal[2];
            }

            if (!display_colors.empty()) {
                if (display_colors.size() == 1) {
                    v.color[0] = display_colors[0][0];
                    v.color[1] = display_colors[0][1];
                    v.color[2] = display_colors[0][2];
                } else if (display_colors.size() == points.size()) {
                    v.color[0] = display_colors[pt_idx][0];
                    v.color[1] = display_colors[pt_idx][1];
                    v.color[2] = display_colors[pt_idx][2];
                } else if (display_colors.size() == face_vertex_counts.size()) {
                    v.color[0] = display_colors[face][0];
                    v.color[1] = display_colors[face][1];
                    v.color[2] = display_colors[face][2];
                } else {
                    v.color[0] = 1.0f;
                    v.color[1] = 1.0f;
                    v.color[2] = 1.0f;
                }
            } else {
                v.color[0] = 1.0f;
                v.color[1] = 1.0f;
                v.color[2] = 1.0f;
            }

            if (!uvs.empty()) {
                if (uvs.size() == face_vertex_indices.size()) {
                    v.uv[0] = uvs[fv_offset + j][0];
                    v.uv[1] = uvs[fv_offset + j][1];
                } else if (uvs.size() == points.size()) {
                    v.uv[0] = uvs[pt_idx][0];
                    v.uv[1] = uvs[pt_idx][1];
                }
            }

            vertices.push_back(v);
        }

        for (int j = 0; j < count - 2; j++) {
            indices.push_back(base);
            indices.push_back(base + j + 1);
            indices.push_back(base + j + 2);
        }

        fv_offset += count;
    }

    if (vertices.empty()) return std::nullopt;

    return AdapterResult{std::move(vertices), std::move(indices)};
}

}  // namespace pts::rendering
