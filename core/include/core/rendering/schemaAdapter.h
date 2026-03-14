#pragma once

#include <core/rendering/vertex.h>
#include <pxr/usd/usd/prim.h>

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <glm/glm.hpp>

namespace pts::rendering {

/// Adapter produced a renderable mesh.
struct MeshResult {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

/// Adapter produced a light source.
struct LightResult {
    enum class Type { Distant, Sphere, Rect, Disk, Dome };
    Type type;
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    float intensity{1.0f};

    // Distant light
    glm::vec3 direction{0.0f, -1.0f, 0.0f};

    // Area/point lights
    float radius{0.0f};
    float width{1.0f};
    float height{1.0f};
};

using AdapterResult = std::variant<MeshResult, LightResult>;

class ISchemaAdapter {
   public:
    virtual ~ISchemaAdapter() = default;
    [[nodiscard]] virtual bool can_adapt(const pxr::UsdPrim& prim) const = 0;
    [[nodiscard]] virtual std::optional<AdapterResult> adapt(const pxr::UsdPrim& prim) const = 0;
};

}  // namespace pts::rendering
