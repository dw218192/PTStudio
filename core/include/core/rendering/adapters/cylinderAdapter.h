#pragma once

#include <core/rendering/schemaAdapter.h>

namespace pts::rendering {

class CylinderAdapter final : public ISchemaAdapter {
   public:
    static const CylinderAdapter& instance();

    bool can_adapt(const pxr::UsdPrim& prim) const override;
    std::optional<AdapterResult> adapt(const pxr::UsdPrim& prim) const override;

   private:
    CylinderAdapter() = default;
};

}  // namespace pts::rendering
