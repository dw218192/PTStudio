#pragma once

#include <core/rendering/webgpu/buffer.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

namespace pts::rendering {

struct Vertex {
    float position[3];
    float normal[3];
    float color[3];
};

struct Mesh {
    webgpu::Buffer vertex_buffer;
    webgpu::Buffer index_buffer;
    uint32_t index_count;
};

struct RenderObject {
    uint32_t mesh_index;
    glm::mat4 transform;
};

struct RenderWorld {
    std::vector<Mesh> meshes;
    std::vector<RenderObject> objects;
    void clear();
};

}  // namespace pts::rendering
