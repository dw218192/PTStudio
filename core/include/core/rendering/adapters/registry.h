#pragma once

#include <core/rendering/adapters/capsuleAdapter.h>
#include <core/rendering/adapters/coneAdapter.h>
#include <core/rendering/adapters/cubeAdapter.h>
#include <core/rendering/adapters/cylinderAdapter.h>
#include <core/rendering/adapters/materialAdapter.h>
#include <core/rendering/adapters/meshAdapter.h>
#include <core/rendering/adapters/sphereAdapter.h>
#include <core/rendering/adapters/transformAdapter.h>
#include <core/rendering/propertyAdapter.h>
#include <core/rendering/schemaAdapter.h>

#include <array>

namespace pts::rendering {

inline constexpr auto k_schema_adapters_build = [](auto&&... adapters) {
    return std::array<const ISchemaAdapter*, sizeof...(adapters)>{&adapters...};
};

inline const auto& k_schema_adapters() {
    static const auto adapters = k_schema_adapters_build(
        MeshAdapter::instance(), CubeAdapter::instance(), SphereAdapter::instance(),
        CylinderAdapter::instance(), ConeAdapter::instance(), CapsuleAdapter::instance());
    return adapters;
}

inline constexpr auto k_property_adapters_build = [](auto&&... adapters) {
    return std::array<const IPropertyAdapter*, sizeof...(adapters)>{&adapters...};
};

// Property adapters run before typed adaptation (tessellation).
// Future: VisibilityAdapter, PurposeAdapter
inline const auto& k_property_adapters() {
    static const auto adapters =
        k_property_adapters_build(TransformAdapter::instance(), MaterialAdapter::instance());
    return adapters;
}

}  // namespace pts::rendering
