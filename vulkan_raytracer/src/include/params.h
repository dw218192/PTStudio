#pragma once
#include <glm/glm.hpp>

namespace PTS {

static constexpr size_t k_max_instances = 100; // max number of meshes in the scene
// each mesh corresponds to a bottom-level acceleration structure instance

static constexpr int k_max_width = 4000; // max width of the output image
static constexpr int k_max_height = 4000; // max height of the output image

struct CameraData {
    glm::mat4 inv_view_proj;
    glm::vec3 cam_pos;
};
struct VertexData {
    glm::vec3 position;
};

} // namespace PTS