#pragma once

#include <core/rendering/vertex.h>
#include <pxr/usd/usd/prim.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

namespace pts {
namespace webgpu {
class Device;
}
namespace rendering {

struct RenderWorld;
class SyncScope;

glm::mat4 compute_world_transform(pxr::UsdPrim prim);

uint32_t resolve_material(pxr::UsdPrim prim, SyncScope& scope);

void upload_mesh(SyncScope& scope, const webgpu::Device* device,
                 const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices,
                 uint32_t mesh_slot);

/// Common sync logic for mesh-producing adapters. Handles transform, material,
/// slot lookup/insert, and GPU upload via the SyncScope.
void sync_object(pxr::UsdPrim prim, SyncScope& scope, const webgpu::Device* device,
                 std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);

struct LightSlot;

/// Common sync logic for light-producing adapters. Handles slot
/// lookup/insert and field population via the SyncScope.
void sync_light(pxr::UsdPrim prim, SyncScope& scope, const LightSlot& light);

}  // namespace rendering
}  // namespace pts
