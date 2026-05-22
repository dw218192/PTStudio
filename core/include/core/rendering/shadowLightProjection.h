#pragma once

#include <core/rendering/renderWorld.h>

#include <cstdint>
#include <glm/glm.hpp>

namespace pts::rendering {

/// Projection metadata for a shadow-casting light. Mirrors the CPU-side
/// inputs used to populate ShadowInfo (see renderWorld.h) and the runtime
/// sampling math in core/shaders/shadow/shadow_sampling.slang.
///
/// `light_size_uv` semantics (matching the shader):
///   * Ortho (distant): tan(half_angle) / ortho_width
///   * Perspective (rect/disk): light_radius / (2 * tan(fov/2))
struct LightProjection {
    glm::mat4 vp{1.0f};
    float near_plane = 0.0f;
    float far_plane = 0.0f;
    float light_size_uv = 0.0f;
    uint32_t projection_type = 0;  // 0 = ortho, 1 = perspective
};

/// Orthographic VP fit to the scene AABB in light space. `light.angle`
/// is in degrees (UsdLuxDistantLight.inputs:angle).
LightProjection compute_distant_light_vp(const LightData& light, const glm::vec3& aabb_min,
                                         const glm::vec3& aabb_max);

/// Perspective VP (90 deg FOV, 1:1 aspect) from the light's position along
/// local -Z. `light.radius` (disk) or `light.width`/`light.height` (rect)
/// feed light_size_uv.
LightProjection compute_area_light_vp(const LightData& light, const glm::vec3& aabb_min,
                                      const glm::vec3& aabb_max);

/// Reconstruct light-space linear depth from stored NDC z in [0, 1].
/// Matches the shader's `linearize_depth` and assumes GLM_FORCE_DEPTH_ZERO_TO_ONE.
float linearize_shadow_depth(float ndc_z, float near_plane, float far_plane,
                             uint32_t projection_type);

}  // namespace pts::rendering
