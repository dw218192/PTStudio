#pragma once

#include <pxr/usd/usd/prim.h>

namespace pts::rendering {

struct RenderObject;
struct Light;
struct RenderWorld;

enum class AdapterAction { Continue, Skip };

class IPropertyAdapter {
   public:
    virtual ~IPropertyAdapter() = default;
    [[nodiscard]] virtual AdapterAction apply(const pxr::UsdPrim& prim, RenderObject& obj,
                                              RenderWorld& world) const = 0;
    [[nodiscard]] virtual AdapterAction apply(const pxr::UsdPrim& prim, Light& light,
                                              RenderWorld& world) const = 0;
};

}  // namespace pts::rendering
