#pragma once

#include <core/rendering/vertex.h>
#include <pxr/usd/usd/prim.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

namespace pts::rendering {

struct RenderWorld;
class SyncScope;

glm::mat4 compute_world_transform(pxr::UsdPrim prim);

uint32_t resolve_material(pxr::UsdPrim prim, SyncScope& scope);

void store_mesh(SyncScope& scope, const std::vector<Vertex>& vertices,
                const std::vector<uint32_t>& indices, uint32_t mesh_slot);

/// Common sync logic for mesh-producing adapters. Handles transform, material,
/// slot lookup/insert, and CPU mesh storage via the SyncScope.
void sync_object(pxr::UsdPrim prim, SyncScope& scope, std::vector<Vertex>& vertices,
                 std::vector<uint32_t>& indices);

struct LightData;

/// Common sync logic for light-producing adapters. Handles slot
/// lookup/insert and field population via the SyncScope.
void sync_light(pxr::UsdPrim prim, SyncScope& scope, const LightData& light);

}  // namespace pts::rendering
