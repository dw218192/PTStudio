#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/materialAdapter.h>
#include <core/rendering/renderWorld.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/shader.h>

namespace pts::rendering {

bool MaterialAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdShadeMaterial>() || prim.IsA<pxr::UsdShadeShader>();
}

void MaterialAdapter::sync(pxr::UsdPrim prim, SyncScope& scope) {
    // Normalize: if this is a shader prim, get the parent material
    pxr::UsdShadeMaterial mat_prim;
    if (prim.IsA<pxr::UsdShadeMaterial>()) {
        mat_prim = pxr::UsdShadeMaterial(prim);
    } else if (prim.IsA<pxr::UsdShadeShader>()) {
        auto parent = prim.GetParent();
        if (parent && parent.IsA<pxr::UsdShadeMaterial>()) {
            mat_prim = pxr::UsdShadeMaterial(parent);
        }
    }
    if (!mat_prim) return;

    auto mat_path = mat_prim.GetPath().GetString();
    auto& cache = scope.material_cache();
    auto it = cache.find(mat_path);
    if (it == cache.end())
        return;  // material not yet in cache -- will be resolved on geometry sync

    // Re-read properties and texture connections from the UsdPreviewSurface shader
    auto surface = mat_prim.ComputeSurfaceSource();
    if (!surface) return;

    pxr::TfToken shader_id;
    surface.GetShaderId(&shader_id);
    if (shader_id != pxr::TfToken("UsdPreviewSurface")) return;

    scope.materials()[it->second] = read_preview_surface(surface, scope);
}

}  // namespace pts::rendering
