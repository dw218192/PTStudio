#pragma once

#include <core/rendering/schemaAdapter.h>

namespace pts::rendering {

class ConeAdapter final : public ISchemaAdapter {
   public:
    static const ConeAdapter& instance();

    bool can_adapt(const pxr::UsdPrim& prim) const override;
    std::optional<AdapterResult> adapt(const pxr::UsdPrim& prim) const override;

   private:
    ConeAdapter() = default;
};

}  // namespace pts::rendering
