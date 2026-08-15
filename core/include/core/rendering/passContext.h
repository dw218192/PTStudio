#pragma once

#include <core/rendering/upAxis.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <glm/glm.hpp>

namespace pts {

namespace webgpu {
class Device;
}

namespace rendering {

class OrbitCamera;
struct RenderWorld;

struct PassContext {
    const webgpu::Device& device;
    WGPUQueue queue;
    const OrbitCamera& camera;
    RenderWorld& world;
    uint32_t viewport_width;
    uint32_t viewport_height;
    glm::mat4 view_matrix;
    glm::mat4 proj_matrix;
    glm::vec3 camera_position;
    float time = 0.0f;
    uint32_t frame_index = 0;
    uint32_t selected_picking_id = UINT32_MAX;
    float meters_per_unit = 0.01f;
    UpAxis up_axis = UpAxis::Y;
};

}  // namespace rendering
}  // namespace pts
