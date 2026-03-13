#pragma once

#include <core/rendering/schemaAdapter.h>

namespace pts::rendering {

class MeshAdapter final : public ISchemaAdapter {
public:
    static const MeshAdapter& instance();

    bool can_adapt(const pxr::UsdPrim& prim) const override;
    std::optional<AdapterResult> adapt(const pxr::UsdPrim& prim) const override;

private:
    MeshAdapter() = default;
};

}  // namespace pts::rendering
