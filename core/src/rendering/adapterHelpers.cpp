#include <core/rendering/adapterHelpers.h>
#include <core/rendering/renderWorld.h>
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

void store_mesh(SyncScope& scope, const std::vector<Vertex>& vertices,
                const std::vector<uint32_t>& indices, uint32_t mesh_slot) {
    auto w = scope.write_mesh(mesh_slot);
    w->cpu_vertices.assign(vertices.begin(), vertices.end());
    w->cpu_indices.assign(indices.begin(), indices.end());
    w->index_count = static_cast<uint32_t>(indices.size());
}

void sync_object(pxr::UsdPrim prim, SyncScope& scope, std::vector<Vertex>& vertices,
                 std::vector<uint32_t>& indices) {
    auto& world = scope.world();
    auto sdf_path = prim.GetPath();
    auto transform = compute_world_transform(prim);
    auto material_index = resolve_material(prim, scope);

    int existing = world.find_object_by_prim(sdf_path.GetText());
    if (existing >= 0) {
        auto w = scope.write_object(static_cast<uint32_t>(existing));
        auto mesh_index = w->mesh_index;
        w->transform = transform;
        w->material_index = material_index;
        store_mesh(scope, vertices, indices, mesh_index);
    } else {
        auto mesh_slot = scope.alloc_mesh_slot();
        auto obj_slot = scope.alloc_object_slot();
        store_mesh(scope, vertices, indices, mesh_slot);
        {
            auto w = scope.write_object(obj_slot);
            w->mesh_index = mesh_slot;
            w->transform = transform;
            w->material_index = material_index;
        }
        scope.set_prim_path(obj_slot, PrimSlot::Kind::Object, sdf_path.GetString());
    }
}

void sync_light(pxr::UsdPrim prim, SyncScope& scope, const LightData& light) {
    auto& world = scope.world();
    auto sdf_path = prim.GetPath();

    int existing = world.find_light_by_prim(sdf_path.GetText());
    if (existing >= 0) {
        auto w = scope.write_light(static_cast<uint32_t>(existing));
        *w = light;
    } else {
        auto slot = scope.alloc_light_slot();
        {
            auto w = scope.write_light(slot);
            *w = light;
        }
        scope.set_prim_path(slot, PrimSlot::Kind::Light, sdf_path.GetString());
    }
}

}  // namespace pts::rendering
