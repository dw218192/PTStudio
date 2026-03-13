#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/xformable.h>

namespace pts::rendering {

void populate_from_stage(RenderWorld& world, const pxr::UsdStageRefPtr& stage,
                         const webgpu::Device& device) {
    for (const auto& prim : pxr::UsdPrimRange(stage->GetPseudoRoot())) {
        if (!prim.IsA<pxr::UsdGeomMesh>()) continue;

        pxr::UsdGeomMesh mesh(prim);

        pxr::VtVec3fArray points;
        mesh.GetPointsAttr().Get(&points);
        if (points.empty()) continue;

        pxr::VtVec3fArray normals;
        mesh.GetNormalsAttr().Get(&normals);

        pxr::VtIntArray face_vertex_counts;
        mesh.GetFaceVertexCountsAttr().Get(&face_vertex_counts);

        pxr::VtIntArray face_vertex_indices;
        mesh.GetFaceVertexIndicesAttr().Get(&face_vertex_indices);

        // Read displayColor
        auto primvars_api = pxr::UsdGeomPrimvarsAPI(prim);

        pxr::VtVec3fArray display_colors;
        auto color_pv = primvars_api.GetPrimvar(pxr::TfToken("displayColor"));
        if (color_pv) {
            color_pv.Get(&display_colors);
        }

        // Read UVs (primvars:st)
        pxr::VtVec2fArray uvs;
        auto uv_pv = primvars_api.GetPrimvar(pxr::TfToken("st"));
        if (uv_pv) {
            uv_pv.Get(&uvs);
        }

        // Build vertex and index arrays
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        size_t fv_offset = 0;
        for (size_t face = 0; face < face_vertex_counts.size(); face++) {
            int count = face_vertex_counts[face];
            if (count < 3) {
                fv_offset += count;
                continue;
            }

            // Compute face normal from first triangle if normals not provided
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

                // Normals
                if (!normals.empty()) {
                    if (normals.size() == face_vertex_indices.size()) {
                        // faceVarying interpolation
                        v.normal[0] = normals[fv_offset + j][0];
                        v.normal[1] = normals[fv_offset + j][1];
                        v.normal[2] = normals[fv_offset + j][2];
                    } else if (normals.size() == points.size()) {
                        // vertex interpolation
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

                // Color
                if (!display_colors.empty()) {
                    if (display_colors.size() == 1) {
                        // constant
                        v.color[0] = display_colors[0][0];
                        v.color[1] = display_colors[0][1];
                        v.color[2] = display_colors[0][2];
                    } else if (display_colors.size() == points.size()) {
                        // per-vertex
                        v.color[0] = display_colors[pt_idx][0];
                        v.color[1] = display_colors[pt_idx][1];
                        v.color[2] = display_colors[pt_idx][2];
                    } else if (display_colors.size() == face_vertex_counts.size()) {
                        // per-face
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

                // UVs
                if (!uvs.empty()) {
                    if (uvs.size() == face_vertex_indices.size()) {
                        // faceVarying interpolation
                        v.uv[0] = uvs[fv_offset + j][0];
                        v.uv[1] = uvs[fv_offset + j][1];
                    } else if (uvs.size() == points.size()) {
                        // vertex interpolation
                        v.uv[0] = uvs[pt_idx][0];
                        v.uv[1] = uvs[pt_idx][1];
                    }
                }

                vertices.push_back(v);
            }

            // Fan triangulation
            for (int j = 0; j < count - 2; j++) {
                indices.push_back(base);
                indices.push_back(base + j + 1);
                indices.push_back(base + j + 2);
            }

            fv_offset += count;
        }

        if (vertices.empty()) continue;

        // Create GPU buffers
        auto vertex_buf = device.create_buffer(
            vertices.size() * sizeof(Vertex),
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst));
        wgpuQueueWriteBuffer(device.queue(), vertex_buf.handle(), 0, vertices.data(),
                             vertices.size() * sizeof(Vertex));

        auto index_buf = device.create_buffer(
            indices.size() * sizeof(uint32_t),
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst));
        wgpuQueueWriteBuffer(device.queue(), index_buf.handle(), 0, indices.data(),
                             indices.size() * sizeof(uint32_t));

        uint32_t mesh_index = static_cast<uint32_t>(world.meshes.size());

        Mesh gpu_mesh;
        gpu_mesh.vertex_buffer = std::move(vertex_buf);
        gpu_mesh.index_buffer = std::move(index_buf);
        gpu_mesh.index_count = static_cast<uint32_t>(indices.size());
        world.meshes.push_back(std::move(gpu_mesh));

        // Compute world transform
        pxr::GfMatrix4d xf =
            pxr::UsdGeomXformable(prim).ComputeLocalToWorldTransform(pxr::UsdTimeCode::Default());
        glm::mat4 transform;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) transform[j][i] = static_cast<float>(xf[i][j]);

        RenderObject obj;
        obj.mesh_index = mesh_index;
        obj.transform = transform;
        obj.prim_path = prim.GetPath().GetString();
        world.objects.push_back(std::move(obj));
    }
}

}  // namespace pts::rendering
