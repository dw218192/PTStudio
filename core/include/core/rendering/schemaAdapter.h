#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <core/rendering/vertex.h>
#include <pxr/usd/usd/prim.h>

namespace pts::rendering {

struct AdapterResult {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

class ISchemaAdapter {
public:
    virtual ~ISchemaAdapter() = default;
    virtual bool can_adapt(const pxr::UsdPrim& prim) const = 0;
    virtual std::optional<AdapterResult> adapt(const pxr::UsdPrim& prim) const = 0;
};

}  // namespace pts::rendering
