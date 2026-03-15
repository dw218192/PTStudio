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

glm::mat4 compute_world_transform(pxr::UsdPrim prim);

uint32_t resolve_material(pxr::UsdPrim prim, RenderWorld& world);

void upload_mesh(RenderWorld& world, const webgpu::Device& device,
                 const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices,
                 uint32_t mesh_slot);

/// Common sync logic for mesh-producing adapters. Handles transform, material,
/// slot lookup/insert, GPU upload, and version bump. Adapters only need to
/// tessellate and call this.
void sync_object(pxr::UsdPrim prim, RenderWorld& world, const webgpu::Device& device,
                 std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);

}  // namespace rendering
}  // namespace pts
