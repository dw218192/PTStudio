#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/euler_angles.hpp>

namespace pts::editor {

struct TransformComponents {
    glm::vec3 translate{0.f};
    glm::vec3 rotate_degrees{0.f};  // euler XYZ in degrees
    glm::vec3 scale{1.f};
};

/// Decompose a 4x4 TRS matrix into translate, rotate (euler XYZ degrees), scale.
inline TransformComponents decompose_trs(const glm::mat4& m) {
    TransformComponents c;
    c.translate = glm::vec3(m[3]);
    c.scale = glm::vec3(glm::length(glm::vec3(m[0])), glm::length(glm::vec3(m[1])),
                        glm::length(glm::vec3(m[2])));

    // Detect reflection (negative determinant) and attribute it to X scale
    if (glm::determinant(glm::mat3(m)) < 0.f) c.scale.x = -c.scale.x;

    glm::mat4 rot_mat(1.f);
    if (c.scale.x != 0.f) rot_mat[0] = m[0] / c.scale.x;
    if (c.scale.y != 0.f) rot_mat[1] = m[1] / c.scale.y;
    if (c.scale.z != 0.f) rot_mat[2] = m[2] / c.scale.z;

    float ex, ey, ez;
    glm::extractEulerAngleXYZ(rot_mat, ex, ey, ez);
    c.rotate_degrees = glm::degrees(glm::vec3(ex, ey, ez));
    return c;
}

/// Compose a 4x4 TRS matrix from translate, rotate (euler XYZ degrees), scale.
inline glm::mat4 compose_trs(const TransformComponents& c) {
    return glm::translate(glm::mat4(1.f), c.translate) *
           glm::eulerAngleXYZ(glm::radians(c.rotate_degrees.x), glm::radians(c.rotate_degrees.y),
                              glm::radians(c.rotate_degrees.z)) *
           glm::scale(glm::mat4(1.f), c.scale);
}

}  // namespace pts::editor
