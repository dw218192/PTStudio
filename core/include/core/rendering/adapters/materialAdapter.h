#pragma once

#include <core/rendering/propertyAdapter.h>

namespace pts::rendering {

class MaterialAdapter final : public IPropertyAdapter {
   public:
    static const MaterialAdapter& instance() {
        static MaterialAdapter s;
        return s;
    }

    [[nodiscard]] AdapterAction apply(const pxr::UsdPrim& prim, RenderObject& obj,
                                      RenderWorld& world) const override;
    [[nodiscard]] AdapterAction apply(const pxr::UsdPrim& prim, Light& light,
                                      RenderWorld& world) const override;

   private:
    MaterialAdapter() = default;
};

}  // namespace pts::rendering
