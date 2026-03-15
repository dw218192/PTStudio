#pragma once

#include <core/rendering/schemaAdapter.h>

namespace pts::rendering {

class SphereAdapter final : public ISchemaAdapter {
   public:
    static const SphereAdapter& instance();

    bool can_adapt(const pxr::UsdPrim& prim) const override;
    std::optional<AdapterResult> adapt(const pxr::UsdPrim& prim) const override;

   private:
    SphereAdapter() = default;
};

}  // namespace pts::rendering
