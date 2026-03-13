#include <core/rendering/adapters/registry.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/xformable.h>

namespace pts::rendering {

void populate_from_stage(RenderWorld& world, const pxr::UsdStageRefPtr& stage,
                         const webgpu::Device& device) {
    for (const auto& prim : pxr::UsdPrimRange(stage->GetPseudoRoot())) {
        for (const auto* adapter : k_schema_adapters()) {
            if (!adapter->can_adapt(prim)) continue;

            auto result = adapter->adapt(prim);
            if (!result) continue;

            auto& [vertices, indices] = *result;

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
            pxr::GfMatrix4d xf = pxr::UsdGeomXformable(prim).ComputeLocalToWorldTransform(
                pxr::UsdTimeCode::Default());
            glm::mat4 transform;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) transform[j][i] = static_cast<float>(xf[i][j]);

            RenderObject obj;
            obj.mesh_index = mesh_index;
            obj.transform = transform;
            obj.prim_path = prim.GetPath().GetString();
            world.objects.push_back(std::move(obj));

            break;  // first matching adapter wins
        }
    }
}

}  // namespace pts::rendering
