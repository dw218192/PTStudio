#pragma once

#include <core/rendering/webgpu/buffer.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace pts::rendering {

struct Vertex {
    float position[3];
    float normal[3];
    float color[3];
    float uv[2];
};

struct Mesh {
    webgpu::Buffer vertex_buffer;
    webgpu::Buffer index_buffer;
    uint32_t index_count;
};

struct RenderObject {
    uint32_t mesh_index;
    glm::mat4 transform;
    std::string prim_path;
};

struct RenderWorld {
    std::vector<Mesh> meshes;
    std::vector<RenderObject> objects;
    void clear();
};

}  // namespace pts::rendering
