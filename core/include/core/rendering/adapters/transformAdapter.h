#pragma once

#include <core/rendering/propertyAdapter.h>

namespace pts::rendering {

class TransformAdapter final : public IPropertyAdapter {
   public:
    static const TransformAdapter& instance() {
        static TransformAdapter s;
        return s;
    }

    [[nodiscard]] AdapterAction apply(const pxr::UsdPrim& prim, RenderObject& obj,
                                      RenderWorld& world) const override;
    [[nodiscard]] AdapterAction apply(const pxr::UsdPrim& prim, Light& light,
                                      RenderWorld& world) const override;

   private:
    TransformAdapter() = default;
};

}  // namespace pts::rendering
