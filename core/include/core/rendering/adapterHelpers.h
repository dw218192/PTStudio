#pragma once

#include <core/rendering/vertex.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdShade/shader.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

namespace pts::rendering {

struct Material;
struct RenderWorld;
class SyncScope;

glm::mat4 compute_world_transform(pxr::UsdPrim prim);

/// Read scalar values and texture connections from a UsdPreviewSurface shader.
/// Resolves UsdUVTexture file paths and loads them via SyncScope::load_texture().
Material read_preview_surface(pxr::UsdShadeShader surface, SyncScope& scope);

uint32_t resolve_material(pxr::UsdPrim prim, SyncScope& scope);

void store_mesh(SyncScope& scope, const std::vector<Vertex>& vertices,
                const std::vector<uint32_t>& indices, uint32_t mesh_slot);

/// Common sync logic for mesh-producing adapters. Handles transform, material,
/// slot lookup/insert, and CPU mesh storage via the SyncScope.
void sync_object(pxr::UsdPrim prim, SyncScope& scope, std::vector<Vertex>& vertices,
                 std::vector<uint32_t>& indices);

/// Overload for subset objects: uses geom_prim for transform/visibility,
/// obj_path for slot identity, and the caller-provided material_index.
void sync_object(pxr::UsdPrim geom_prim, const pxr::SdfPath& obj_path, uint32_t material_index,
                 SyncScope& scope, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);

struct LightData;
struct CameraData;

/// Common sync logic for light-producing adapters. Handles slot
/// lookup/insert and field population via the SyncScope.
void sync_light(pxr::UsdPrim prim, SyncScope& scope, const LightData& light);

/// Common sync logic for camera-producing adapters. Handles slot
/// lookup/insert and field population via the SyncScope.
void sync_camera(pxr::UsdPrim prim, SyncScope& scope, const CameraData& camera);

// --- Proxy mesh geometry generators (exposed for testing) ---

/// Generate a 2-triangle quad centered at origin, facing -Z.
void generate_rect_mesh(float width, float height, std::vector<Vertex>& out_vertices,
                        std::vector<uint32_t>& out_indices);

/// Generate a triangle-fan disk centered at origin, facing -Z.
void generate_disk_mesh(float radius, std::vector<Vertex>& out_vertices,
                        std::vector<uint32_t>& out_indices);

/// Generate a UV sphere centered at origin, normals pointing outward.
void generate_sphere_mesh(float radius, std::vector<Vertex>& out_vertices,
                          std::vector<uint32_t>& out_indices);

}  // namespace pts::rendering
