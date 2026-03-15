#include <core/rendering/adapters/transformAdapter.h>
#include <core/rendering/renderWorld.h>
#include <pxr/usd/usdGeom/xformable.h>

namespace pts::rendering {

namespace {

glm::mat4 compute_world_transform(const pxr::UsdPrim& prim) {
    pxr::GfMatrix4d xf =
        pxr::UsdGeomXformable(prim).ComputeLocalToWorldTransform(pxr::UsdTimeCode::Default());
    glm::mat4 transform;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) transform[i][j] = static_cast<float>(xf[i][j]);
    return transform;
}

}  // namespace

AdapterAction TransformAdapter::apply(const pxr::UsdPrim& prim, RenderObject& obj,
                                      RenderWorld& /*world*/) const {
    obj.transform = compute_world_transform(prim);
    return AdapterAction::Continue;
}

AdapterAction TransformAdapter::apply(const pxr::UsdPrim& prim, Light& light,
                                      RenderWorld& /*world*/) const {
    light.transform = compute_world_transform(prim);
    return AdapterAction::Continue;
}

}  // namespace pts::rendering
