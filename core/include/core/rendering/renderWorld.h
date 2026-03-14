#pragma once

#include <core/rendering/vertex.h>
#include <core/rendering/webgpu/buffer.h>

#include <climits>
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace pts::rendering {

static constexpr uint32_t k_no_material = UINT32_MAX;

struct Material {
    glm::vec3 diffuse_color{1.0f, 1.0f, 1.0f};
    float metallic{0.0f};
    float roughness{0.5f};
    float opacity{1.0f};
    uint32_t _padding[2]{};
};
static_assert(sizeof(Material) == 32, "Material must be 32 bytes for GPU alignment");

struct Mesh {
    webgpu::Buffer vertex_buffer;
    webgpu::Buffer index_buffer;
    uint32_t index_count;
};

struct RenderObject {
    uint32_t mesh_index;
    uint32_t material_index{k_no_material};
    glm::mat4 transform;
    std::string prim_path;
};

struct Light {
    enum class Type { Distant, Sphere, Rect, Disk, Dome };
    Type type;
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    float intensity{1.0f};
    glm::mat4 transform;
    std::string prim_path;

    // Distant light
    glm::vec3 direction{0.0f, -1.0f, 0.0f};

    // Area/point lights
    float radius{0.0f};
    float width{1.0f};
    float height{1.0f};
};

struct RenderWorld {
    std::vector<Mesh> meshes;
    std::vector<RenderObject> objects;
    std::vector<Material> materials;
    std::vector<Light> lights;
    std::unordered_map<std::string, uint32_t> material_cache;
    void clear();
};

}  // namespace pts::rendering
