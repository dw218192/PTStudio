#pragma once

#include <core/rendering/propertyDescriptor.h>
#include <pxr/usd/usd/prim.h>

#include <vector>

namespace pts {
namespace webgpu {
class Device;
}
namespace rendering {
class SyncScope;

class ISceneAdapter {
   public:
    virtual ~ISceneAdapter() = default;
    [[nodiscard]] virtual bool can_adapt(const pxr::UsdPrim& prim) const = 0;
    virtual void sync(pxr::UsdPrim prim, SyncScope& scope, const webgpu::Device& device) = 0;
    virtual std::vector<PropertyDescriptor> get_properties(const pxr::UsdPrim& prim) const {
        return {};
    }
};

}  // namespace rendering
}  // namespace pts
