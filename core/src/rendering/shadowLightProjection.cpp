#include <core/diagnostics.h>
#include <core/rendering/shadowLightProjection.h>

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>

namespace pts::rendering {

LightProjection compute_distant_light_vp(const LightData& light, const glm::vec3& aabb_min,
                                         const glm::vec3& aabb_max) {
    auto dir = glm::normalize(light.direction);

    auto center = (aabb_min + aabb_max) * 0.5f;
    auto half_diag = glm::length(aabb_max - aabb_min) * 0.5f;

    auto up = glm::vec3(0, 1, 0);
    if (std::abs(glm::dot(dir, up)) > 0.99f) up = glm::vec3(1, 0, 0);

    auto light_view = glm::lookAt(center - dir * half_diag, center, up);

    glm::vec3 ls_min(std::numeric_limits<float>::max());
    glm::vec3 ls_max(std::numeric_limits<float>::lowest());
    for (int c = 0; c < 8; ++c) {
        glm::vec3 corner((c & 1) ? aabb_max.x : aabb_min.x, (c & 2) ? aabb_max.y : aabb_min.y,
                         (c & 4) ? aabb_max.z : aabb_min.z);
        glm::vec3 ls_pt = glm::vec3(light_view * glm::vec4(corner, 1.0f));
        ls_min = glm::min(ls_min, ls_pt);
        ls_max = glm::max(ls_max, ls_pt);
    }

    float near_plane = -ls_max.z;
    float far_plane = -ls_min.z;
    float ortho_width = ls_max.x - ls_min.x;

    auto ortho_proj = glm::ortho(ls_min.x, ls_max.x, ls_min.y, ls_max.y, near_plane, far_plane);

    // light.angle is in degrees (UsdLuxDistantLight.inputs:angle), matching
    // light_lib.slang's `radians(light.angle * 0.5)` convention.
    float half_angle_rad = glm::radians(std::max(light.angle, 0.0f) * 0.5f);
    float light_size_uv = (ortho_width > 0.0f) ? (std::tan(half_angle_rad) / ortho_width) : 0.0f;
    light_size_uv *= std::max(light.shadow_pcss_softness, 0.0f);

    LightProjection out;
    out.vp = ortho_proj * light_view;
    out.near_plane = near_plane;
    out.far_plane = far_plane;
    out.light_size_uv = light_size_uv;
    out.projection_type = 0;
    return out;
}

LightProjection compute_area_light_vp(const LightData& light, const glm::vec3& aabb_min,
                                      const glm::vec3& aabb_max, uint32_t face) {
    PRECONDITION(face < 6);
    // Six conventional perspective maps retain coverage at grazing angles.
    // The point being approximated stays at the emitter center as it moves.
    static const glm::vec3 directions[] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                           {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    static const glm::vec3 ups[] = {{0, -1, 0}, {0, -1, 0}, {0, 0, 1},
                                    {0, 0, -1}, {0, -1, 0}, {0, -1, 0}};
    glm::vec3 position(light.transform[3]);
    float far_plane = 0.01f;
    for (int c = 0; c < 8; ++c) {
        glm::vec3 corner((c & 1) ? aabb_max.x : aabb_min.x, (c & 2) ? aabb_max.y : aabb_min.y,
                         (c & 4) ? aabb_max.z : aabb_min.z);
        far_plane = std::max(far_plane, glm::length(corner - position));
    }
    float near_plane = std::max(0.001f, far_plane * 0.0001f);
    auto view = glm::lookAt(position, position + directions[face], ups[face]);
    auto proj = glm::perspective(glm::radians(90.0f), 1.0f, near_plane, far_plane);
    LightProjection out;
    out.vp = proj * view;
    out.near_plane = near_plane;
    out.far_plane = far_plane;
    out.projection_type = 2;
    return out;
}

float linearize_shadow_depth(float ndc_z, float near_plane, float far_plane,
                             uint32_t projection_type) {
    if (projection_type == 0u) {
        return near_plane + ndc_z * (far_plane - near_plane);
    }
    float denom = far_plane - ndc_z * (far_plane - near_plane);
    return (near_plane * far_plane) / std::max(denom, 1e-6f);
}

}  // namespace pts::rendering
