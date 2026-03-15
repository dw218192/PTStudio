#pragma once

#include <core/rendering/sceneAdapter.h>

namespace pts::rendering {

class CylinderAdapter final : public ISceneAdapter {
   public:
    static CylinderAdapter& instance();

    bool can_adapt(const pxr::UsdPrim& prim) const override;
    void sync(const pxr::UsdPrim& prim, RenderWorld& world,
              const webgpu::Device& device) override;

   private:
    CylinderAdapter() = default;
};

}  // namespace pts::rendering
