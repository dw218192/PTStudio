#pragma once

#include <core/rendering/sceneAdapter.h>

namespace pts::rendering {

class LightAdapter final : public ISceneAdapter {
   public:
    static LightAdapter& instance();

    bool can_adapt(const pxr::UsdPrim& prim) const override;
    void sync(pxr::UsdPrim prim, SyncScope& scope) override;
    std::vector<PropertyDescriptor> get_properties(const pxr::UsdPrim& prim) const override;

   private:
    LightAdapter() = default;
};

}  // namespace pts::rendering
