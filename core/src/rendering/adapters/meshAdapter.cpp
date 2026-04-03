#include <core/diagnostics.h>
#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/meshAdapter.h>
#include <core/rendering/renderWorld.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/tokens.h>

#include <unordered_map>
#include <unordered_set>

namespace pts::rendering {

namespace {

/// Extract a subset of a triangulated mesh by face indices.
/// Returns compacted vertex and index arrays containing only the triangles
/// whose originating face (from prim_params) is in face_set.
/// prim_params values are encoded via HdMeshUtil::EncodeCoarseFaceParam;
/// the face index is decoded as (param >> 2).
std::pair<std::vector<Vertex>, std::vector<uint32_t>> extract_subset_mesh(
    const std::vector<Vertex>& all_vertices, const std::vector<uint32_t>& all_indices,
    const pxr::VtIntArray& prim_params, const std::unordered_set<int>& face_set) {
    size_t num_triangles = prim_params.size();
    INVARIANT(all_indices.size() == num_triangles * 3);

    std::vector<uint32_t> raw_indices;
    for (size_t t = 0; t < num_triangles; ++t) {
        int face_index = pxr::HdMeshUtil::DecodeFaceIndexFromCoarseFaceParam(prim_params[t]);
        if (face_set.count(face_index)) {
            raw_indices.push_back(all_indices[t * 3 + 0]);
            raw_indices.push_back(all_indices[t * 3 + 1]);
            raw_indices.push_back(all_indices[t * 3 + 2]);
        }
    }

    if (raw_indices.empty()) return {{}, {}};

    std::unordered_map<uint32_t, uint32_t> remap;
    std::vector<Vertex> compact_vertices;
    std::vector<uint32_t> compact_indices;
    compact_indices.reserve(raw_indices.size());

    for (uint32_t idx : raw_indices) {
        auto [it, inserted] =
            remap.try_emplace(idx, static_cast<uint32_t>(compact_vertices.size()));
        if (inserted) {
            compact_vertices.push_back(all_vertices[idx]);
        }
        compact_indices.push_back(it->second);
    }

    return {std::move(compact_vertices), std::move(compact_indices)};
}

}  // namespace

MeshAdapter& MeshAdapter::instance() {
    static MeshAdapter s_instance;
    return s_instance;
}

bool MeshAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomMesh>() || prim.IsA<pxr::UsdGeomSubset>();
}

void MeshAdapter::sync(pxr::UsdPrim prim, SyncScope& scope) {
    // GeomSubset children are handled by the parent mesh's sync — skip them.
    if (prim.IsA<pxr::UsdGeomSubset>()) return;

    pxr::UsdGeomMesh mesh(prim);

    pxr::VtVec3fArray points;
    mesh.GetPointsAttr().Get(&points);
    if (points.empty()) return;

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
    pxr::VtIntArray seq_fv_indices(face_vertex_indices.size());

    size_t fv_offset = 0;
    for (size_t face = 0; face < face_vertex_counts.size(); face++) {
        int count = face_vertex_counts[face];
        if (fv_offset + count > face_vertex_indices.size()) break;

        pxr::GfVec3f face_normal(0, 0, 0);
        if (normals.empty() && count >= 3) {
            int i0 = face_vertex_indices[fv_offset];
            int i1 = face_vertex_indices[fv_offset + 1];
            int i2 = face_vertex_indices[fv_offset + 2];
            pxr::GfVec3f e1 = points[i1] - points[i0];
            pxr::GfVec3f e2 = points[i2] - points[i0];
            face_normal = e1 ^ e2;
            float len = face_normal.GetLength();
            if (len > 0) face_normal /= len;
        }

        for (int j = 0; j < count; j++) {
            int pt_idx = face_vertex_indices[fv_offset + j];
            seq_fv_indices[fv_offset + j] = static_cast<int>(vertices.size());

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

        fv_offset += count;
    }

    if (vertices.empty()) return;

    pxr::TfToken orientation;
    mesh.GetOrientationAttr().Get(&orientation);
    if (orientation.IsEmpty()) orientation = pxr::UsdGeomTokens->leftHanded;

    pxr::HdMeshTopology hd_topo(pxr::PxOsdOpenSubdivTokens->none, orientation, face_vertex_counts,
                                seq_fv_indices);
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

    // --- Check for materialBind GeomSubsets ---
    auto all_subsets = pxr::UsdGeomSubset::GetAllGeomSubsets(mesh);
    std::vector<pxr::UsdGeomSubset> material_subsets;
    for (const auto& subset : all_subsets) {
        pxr::TfToken family;
        subset.GetFamilyNameAttr().Get(&family);
        if (family == pxr::TfToken("materialBind")) {
            material_subsets.push_back(subset);
        }
    }

    if (material_subsets.empty()) {
        sync_object(prim, scope, vertices, indices);
        return;
    }

    // Emit one render object per subset.
    std::vector<bool> face_covered(face_vertex_counts.size(), false);

    for (const auto& subset : material_subsets) {
        pxr::VtIntArray face_indices;
        subset.GetIndicesAttr().Get(&face_indices);

        std::unordered_set<int> face_set;
        face_set.reserve(face_indices.size());
        for (int fi : face_indices) {
            PRECONDITION(fi >= 0 && static_cast<size_t>(fi) < face_vertex_counts.size());
            face_covered[static_cast<size_t>(fi)] = true;
            face_set.insert(fi);
        }

        auto [sub_verts, sub_idxs] = extract_subset_mesh(vertices, indices, prim_params, face_set);
        if (sub_idxs.empty()) continue;

        auto material_index = resolve_material(subset.GetPrim(), scope);
        if (material_index == k_no_material) {
            material_index = k_default_material;
        }

        sync_object(prim, subset.GetPrim().GetPath(), material_index, scope, sub_verts, sub_idxs);
    }

    // Emit a remainder object for faces not covered by any subset,
    // using the mesh-level material binding.
    std::unordered_set<int> remaining_faces;
    for (size_t i = 0; i < face_covered.size(); ++i) {
        if (!face_covered[i]) remaining_faces.insert(static_cast<int>(i));
    }

    if (!remaining_faces.empty()) {
        auto [rem_verts, rem_idxs] =
            extract_subset_mesh(vertices, indices, prim_params, remaining_faces);
        if (!rem_idxs.empty()) {
            sync_object(prim, scope, rem_verts, rem_idxs);
        }
    }
}

}  // namespace pts::rendering
