#pragma once

#include <core/rendering/sceneAdapter.h>

namespace pts::rendering {

class CameraAdapter final : public ISceneAdapter {
   public:
    static CameraAdapter& instance();

    bool can_adapt(const pxr::UsdPrim& prim) const override;
    void sync(pxr::UsdPrim prim, SyncScope& scope) override;
    std::vector<PropertyDescriptor> get_properties(const pxr::UsdPrim& prim) const override;
    std::vector<PrimFactory> get_factories() const override;

   private:
    CameraAdapter() = default;
};

}  // namespace pts::rendering
