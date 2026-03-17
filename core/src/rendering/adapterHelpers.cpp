#include <core/rendering/adapterHelpers.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/webgpu/device.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>

namespace pts::rendering {

glm::mat4 compute_world_transform(pxr::UsdPrim prim) {
    pxr::GfMatrix4d xf =
        pxr::UsdGeomXformable(prim).ComputeLocalToWorldTransform(pxr::UsdTimeCode::Default());
    glm::mat4 transform;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) transform[i][j] = static_cast<float>(xf[i][j]);
    return transform;
}

uint32_t resolve_material(pxr::UsdPrim prim, SyncScope& scope) {
    auto binding = pxr::UsdShadeMaterialBindingAPI(prim).ComputeBoundMaterial();
    if (!binding) {
        return k_no_material;
    }

    auto mat_path = binding.GetPath().GetString();

    auto& cache = scope.material_cache();
    auto it = cache.find(mat_path);
    if (it != cache.end()) {
        return it->second;
    }

    Material mat;

    auto surface = binding.ComputeSurfaceSource();
    if (surface) {
        pxr::TfToken shader_id;
        surface.GetShaderId(&shader_id);
        if (shader_id == pxr::TfToken("UsdPreviewSurface")) {
            if (auto input = surface.GetInput(pxr::TfToken("diffuseColor"))) {
                pxr::GfVec3f color;
                if (input.Get(&color)) mat.diffuse_color = {color[0], color[1], color[2]};
            }
            if (auto input = surface.GetInput(pxr::TfToken("metallic"))) {
                input.Get(&mat.metallic);
            }
            if (auto input = surface.GetInput(pxr::TfToken("roughness"))) {
                input.Get(&mat.roughness);
            }
            if (auto input = surface.GetInput(pxr::TfToken("opacity"))) {
                input.Get(&mat.opacity);
            }
        }
    }

    auto& materials = scope.materials();
    auto index = static_cast<uint32_t>(materials.size());
    materials.push_back(mat);
    cache[mat_path] = index;
    return index;
}

void upload_mesh(SyncScope& scope, const webgpu::Device& device,
                 const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices,
                 uint32_t mesh_slot) {
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

    Mesh& gpu_mesh = scope.mesh(mesh_slot);
    gpu_mesh.vertex_buffer = std::move(vertex_buf);
    gpu_mesh.index_buffer = std::move(index_buf);
    gpu_mesh.index_count = static_cast<uint32_t>(indices.size());
    gpu_mesh.cpu_indices.assign(indices.begin(), indices.end());
    ++gpu_mesh.version;
}

void sync_object(pxr::UsdPrim prim, SyncScope& scope, const webgpu::Device& device,
                 std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
    auto& world = scope.world();
    auto sdf_path = prim.GetPath();
    auto transform = compute_world_transform(prim);
    auto material_index = resolve_material(prim, scope);

    int existing = world.find_object_by_prim(sdf_path.GetText());
    if (existing >= 0) {
        auto& obj = scope.object(static_cast<uint32_t>(existing));
        obj.transform = transform;
        obj.material_index = material_index;
        upload_mesh(scope, device, vertices, indices, obj.mesh_index);
    } else {
        auto prim_path = sdf_path.GetString();
        auto mesh_slot = scope.alloc_mesh_slot();
        auto obj_slot = scope.alloc_object_slot();
        upload_mesh(scope, device, vertices, indices, mesh_slot);
        auto& obj = scope.object(obj_slot);
        obj.mesh_index = mesh_slot;
        obj.transform = transform;
        obj.material_index = material_index;
        obj.prim_path = std::move(prim_path);
        scope.set_prim_slot(obj.prim_path, PrimSlot{PrimSlot::Kind::Object, obj_slot});
    }
}

void sync_light(pxr::UsdPrim prim, SyncScope& scope, const Light& light) {
    auto& world = scope.world();
    auto sdf_path = prim.GetPath();

    int existing = world.find_light_by_prim(sdf_path.GetText());
    if (existing >= 0) {
        auto& dst = scope.light(static_cast<uint32_t>(existing));
        auto prim_path = std::move(dst.prim_path);
        dst = light;
        dst.prim_path = std::move(prim_path);
        ++dst.version;
        scope.mark_light_dirty(static_cast<uint32_t>(existing));
        scope.bump_light_version();
    } else {
        auto slot = scope.alloc_light_slot();
        auto prim_path = sdf_path.GetString();
        auto& dst = scope.light(slot);
        dst = light;
        dst.prim_path = std::move(prim_path);
        scope.set_prim_slot(dst.prim_path, PrimSlot{PrimSlot::Kind::Light, slot});
    }
}

}  // namespace pts::rendering
