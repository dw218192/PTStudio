#include <core/rendering/adapters/materialAdapter.h>
#include <core/rendering/renderWorld.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>

namespace pts::rendering {

AdapterAction MaterialAdapter::apply(const pxr::UsdPrim& prim, RenderObject& obj,
                                     RenderWorld& world) const {
    auto binding = pxr::UsdShadeMaterialBindingAPI(prim).ComputeBoundMaterial();
    if (!binding) {
        obj.material_index = k_no_material;
        return AdapterAction::Continue;
    }

    auto mat_path = binding.GetPath().GetString();

    auto it = world.material_cache.find(mat_path);
    if (it != world.material_cache.end()) {
        obj.material_index = it->second;
        return AdapterAction::Continue;
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

    auto index = static_cast<uint32_t>(world.materials.size());
    world.materials.push_back(mat);
    world.material_cache[mat_path] = index;
    obj.material_index = index;
    return AdapterAction::Continue;
}

AdapterAction MaterialAdapter::apply(const pxr::UsdPrim& /*prim*/, Light& /*light*/,
                                     RenderWorld& /*world*/) const {
    return AdapterAction::Continue;
}

}  // namespace pts::rendering
