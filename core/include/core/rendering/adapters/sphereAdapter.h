#pragma once

#include <core/rendering/sceneAdapter.h>

namespace pts::rendering {

class SphereAdapter final : public ISceneAdapter {
   public:
    static SphereAdapter& instance();

    bool can_adapt(const pxr::UsdPrim& prim) const override;
    void sync(pxr::UsdPrim prim, SyncScope& scope, const webgpu::Device& device) override;

   private:
    SphereAdapter() = default;
};

}  // namespace pts::rendering
