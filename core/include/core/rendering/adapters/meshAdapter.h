#pragma once

#include <core/rendering/sceneAdapter.h>

namespace pts::rendering {

class MeshAdapter final : public ISceneAdapter {
   public:
    static MeshAdapter& instance();

    bool can_adapt(const pxr::UsdPrim& prim) const override;
    void sync(pxr::UsdPrim prim, RenderWorld& world, const webgpu::Device& device) override;

   private:
    MeshAdapter() = default;
};

}  // namespace pts::rendering
