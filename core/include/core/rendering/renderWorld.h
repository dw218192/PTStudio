#pragma once

#include <core/rendering/vertex.h>
#include <core/rendering/webgpu/buffer.h>

#include <climits>
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
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

struct RenderWorld {
    std::vector<Mesh> meshes;
    std::vector<RenderObject> objects;
    std::vector<Material> materials;
    void clear();
};

}  // namespace pts::rendering
