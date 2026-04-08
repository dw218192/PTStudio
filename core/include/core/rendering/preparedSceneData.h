#pragma once

#include <core/rendering/bvh.h>
#include <core/rendering/packedTriangle.h>
#include <core/rendering/renderWorld.h>

#include <cstdint>
#include <vector>

namespace pts::rendering {

/// Pure CPU snapshot of scene data ready for GPU upload.
/// No GPU handles, no WGPUBuffer, no device references.
struct PreparedSceneData {
    // Materials
    std::vector<Material> materials;
    bool materials_dirty = false;

    // Lights
    std::vector<Light> gpu_lights;
    bool lights_dirty = false;
    struct PartialLightUpdate {
        uint32_t gpu_index;
        Light data;
    };
    std::vector<PartialLightUpdate> partial_light_updates;

    // BVH + geometry
    std::vector<BVHNode> all_nodes;
    std::vector<PackedTriangle> all_tris;
    std::vector<GPUInstance> gpu_instances;
    BVH tlas;  // built on worker, swapped to m_tlas on main thread
    uint32_t tlas_node_count = 0;
    uint32_t instance_count = 0;
    bool geometry_dirty = false;

    // Textures — non-owning pixel pointers, stable during frame
    struct TextureLayer {
        const uint16_t* pixels;  // RGBA16Float (half-precision)
        uint32_t width;
        uint32_t height;
    };
    std::vector<TextureLayer> texture_layers;
    uint32_t texture_size = 0;
    bool textures_dirty = false;
};

}  // namespace pts::rendering
