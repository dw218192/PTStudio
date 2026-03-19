#pragma once

#include <core/rendering/propertyDescriptor.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>

#include <string>
#include <vector>

namespace pts::rendering {

class SyncScope;

struct PrimFactory {
    std::string category;
    std::string display_name;
    std::string base_name;
    using DefineFn = pxr::UsdPrim (*)(const pxr::UsdStageRefPtr&, const pxr::SdfPath&);
    DefineFn define = nullptr;
};

class ISceneAdapter {
   public:
    virtual ~ISceneAdapter() = default;
    [[nodiscard]] virtual bool can_adapt(const pxr::UsdPrim& prim) const = 0;
    virtual void sync(pxr::UsdPrim prim, SyncScope& scope) = 0;
    virtual std::vector<PropertyDescriptor> get_properties(const pxr::UsdPrim& prim) const {
        return {};
    }
    virtual std::vector<PrimFactory> get_factories() const {
        return {};
    }
};

}  // namespace pts::rendering
