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
    // light.slang's `radians(light.angle * 0.5)` convention.
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
                                      const glm::vec3& aabb_max) {
    glm::vec3 position(light.transform[3]);
    glm::vec3 forward = glm::normalize(-glm::vec3(light.transform[2]));
    glm::vec3 up = glm::normalize(glm::vec3(light.transform[1]));
    if (std::abs(glm::dot(forward, up)) > 0.99f) {
        up = (std::abs(forward.y) > 0.9f) ? glm::vec3(0.0f, 0.0f, 1.0f)
                                          : glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // Effective world-space light radius (isotropic approximation).
    // For rect lights we take the larger half-extent; this is conservative --
    // the penumbra is actually anisotropic, but representing that needs a 2D
    // light-space oriented kernel (follow-up).
    float light_radius = 0.0f;
    if (light.type == LightData::Type::Disk) {
        light_radius = std::max(light.radius, 0.0f);
    } else if (light.type == LightData::Type::Rect) {
        float hw = std::max(light.width, 0.0f) * 0.5f;
        float hh = std::max(light.height, 0.0f) * 0.5f;
        light_radius = std::max(hw, hh);
    }

    // 120-deg FOV covers +-tan(60)=+-1.73 at unit depth, giving good
    // hemisphere coverage for area light shadows without a cubemap.
    // Compared to 90-deg this loses ~1.7x texel density but eliminates
    // the hard frustum cutoff at grazing angles.
    constexpr float k_fov_y_rad = glm::radians(120.0f);
    float half_tan = std::tan(k_fov_y_rad * 0.5f);

    float far_plane = 0.0f;
    for (int c = 0; c < 8; ++c) {
        glm::vec3 corner((c & 1) ? aabb_max.x : aabb_min.x, (c & 2) ? aabb_max.y : aabb_min.y,
                         (c & 4) ? aabb_max.z : aabb_min.z);
        far_plane = std::max(far_plane, glm::length(corner - position));
    }
    INVARIANT(far_plane > 0.0f);
    float near_plane = std::max(0.001f, far_plane * 0.01f);

    auto light_view = glm::lookAt(position, position + forward, up);
    auto light_proj = glm::perspective(k_fov_y_rad, 1.0f, near_plane, far_plane);

    float light_size_uv = light_radius / (2.0f * half_tan);
    light_size_uv *= std::max(light.shadow_pcss_softness, 0.0f);

    LightProjection out;
    out.vp = light_proj * light_view;
    out.near_plane = near_plane;
    out.far_plane = far_plane;
    out.light_size_uv = light_size_uv;
    out.projection_type = 1;
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
