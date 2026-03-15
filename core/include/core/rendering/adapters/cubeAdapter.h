#pragma once

#include <core/rendering/sceneAdapter.h>

namespace pts::rendering {

class CubeAdapter final : public ISceneAdapter {
   public:
    static CubeAdapter& instance();

    bool can_adapt(const pxr::UsdPrim& prim) const override;
    void sync(pxr::UsdPrim prim, SyncScope& scope, const webgpu::Device& device) override;

   private:
    CubeAdapter() = default;
};

}  // namespace pts::rendering
