#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace pts::rendering {

/// GPU-aligned flattened triangle for ray intersection (128 bytes).
/// Positions and normals are in local space; transform is stored per-instance.
struct PackedTriangle {
    glm::vec3 v0;
    uint32_t _pad0{};
    glm::vec3 v1;
    uint32_t _pad1{};
    glm::vec3 v2;
    uint32_t _pad2{};
    glm::vec3 n0;
    uint32_t _pad3{};
    glm::vec3 n1;
    uint32_t _pad4{};
    glm::vec3 n2;
    uint32_t _pad6{};
    glm::vec2 uv0{};
    glm::vec2 uv1{};
    glm::vec2 uv2{};
    uint32_t _pad5[2]{};
};
static_assert(sizeof(PackedTriangle) == 128, "PackedTriangle must be 128 bytes for GPU alignment");

}  // namespace pts::rendering
