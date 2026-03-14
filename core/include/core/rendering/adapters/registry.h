#pragma once

#include <core/rendering/adapters/capsuleAdapter.h>
#include <core/rendering/adapters/coneAdapter.h>
#include <core/rendering/adapters/cubeAdapter.h>
#include <core/rendering/adapters/cylinderAdapter.h>
#include <core/rendering/adapters/meshAdapter.h>
#include <core/rendering/adapters/sphereAdapter.h>
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

}  // namespace pts::rendering
