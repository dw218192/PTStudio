#pragma once

#include <core/rendering/sceneAdapter.h>

namespace pts::rendering {

class MaterialAdapter final : public ISceneAdapter {
   public:
    static MaterialAdapter& instance() {
        static MaterialAdapter adapter;
        return adapter;
    }

    MaterialAdapter(const MaterialAdapter&) = delete;
    MaterialAdapter& operator=(const MaterialAdapter&) = delete;

    [[nodiscard]] bool can_adapt(const pxr::UsdPrim& prim) const override;
    void sync(pxr::UsdPrim prim, SyncScope& scope) override;

   private:
    MaterialAdapter() = default;
};

}  // namespace pts::rendering
