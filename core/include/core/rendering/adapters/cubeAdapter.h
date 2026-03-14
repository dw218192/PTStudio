#pragma once

#include <core/rendering/schemaAdapter.h>

namespace pts::rendering {

class CubeAdapter final : public ISchemaAdapter {
   public:
    static const CubeAdapter& instance();

    bool can_adapt(const pxr::UsdPrim& prim) const override;
    std::optional<AdapterResult> adapt(const pxr::UsdPrim& prim) const override;

   private:
    CubeAdapter() = default;
};

}  // namespace pts::rendering
