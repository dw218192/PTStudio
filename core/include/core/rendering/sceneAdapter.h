#pragma once

#include <pxr/usd/usd/prim.h>

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
    virtual void sync(pxr::UsdPrim prim, SyncScope& scope,
                      const webgpu::Device& device) = 0;
};

}  // namespace rendering
}  // namespace pts
