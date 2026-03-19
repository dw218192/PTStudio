#pragma once

#include <core/rendering/sceneAdapter.h>

namespace pts::rendering {

class ConeAdapter final : public ISceneAdapter {
   public:
    static ConeAdapter& instance();

    bool can_adapt(const pxr::UsdPrim& prim) const override;
    void sync(pxr::UsdPrim prim, SyncScope& scope) override;
    std::vector<PropertyDescriptor> get_properties(const pxr::UsdPrim& prim) const override;

   private:
    ConeAdapter() = default;
};

}  // namespace pts::rendering
