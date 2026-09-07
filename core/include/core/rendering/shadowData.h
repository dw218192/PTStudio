#pragma once

// Buffer declarations and depth reconstruction shared by C++ and Slang.
// Keep explicit scalar/vector widths: C++ and GPU packing rules can differ.
#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <type_traits>

#define PTS_SHARED
#define PTS_SHARED_FUNCTION inline
#define PTS_DEFAULT(value) {value}
namespace pts::rendering {
using float2 = glm::vec2;
using float4 = glm::vec4;
using float4x4 = glm::mat4;
using uint = std::uint32_t;
#else
#define PTS_SHARED public
#define PTS_SHARED_FUNCTION public
#define PTS_DEFAULT(value)
#endif

// One entry per light; has_shadow == 0 marks inactive entries.
PTS_SHARED struct ShadowInfo {
    PTS_SHARED float4x4 light_vp PTS_DEFAULT(1.0f);
    PTS_SHARED float texel_size PTS_DEFAULT(0);
    PTS_SHARED float normal_bias PTS_DEFAULT(0);
    PTS_SHARED uint has_shadow PTS_DEFAULT(0);
    PTS_SHARED uint layer PTS_DEFAULT(0);
    PTS_SHARED float light_near PTS_DEFAULT(0);
    PTS_SHARED float light_far PTS_DEFAULT(0);
    PTS_SHARED float light_size_uv PTS_DEFAULT(0);
    PTS_SHARED uint projection_type PTS_DEFAULT(0);   // 0 = ortho, 1 = perspective, 2 = cube
    PTS_SHARED float4 light_position PTS_DEFAULT(0);  // xyz = cube origin, w = disk (1) or rect (0)
    PTS_SHARED float4 light_u PTS_DEFAULT(0);         // emitter half-axis, softness included
    PTS_SHARED float4 light_v PTS_DEFAULT(0);
};

PTS_SHARED struct ShadowVisibilityUniforms {
    PTS_SHARED float4x4 inv_view_proj PTS_DEFAULT(1.0f);
    PTS_SHARED float2 viewport_size PTS_DEFAULT(0);
    PTS_SHARED uint shadow_light_index PTS_DEFAULT(0);
    PTS_SHARED uint frame_index PTS_DEFAULT(0);
};

// Reconstruct linear light-space depth from zero-to-one NDC depth.
PTS_SHARED_FUNCTION float linearize_shadow_depth(float ndc_z, float near_plane, float far_plane,
                                                 uint projection_type) {
    if (projection_type == 0u) {
        return near_plane + ndc_z * (far_plane - near_plane);
    }
    float denom = far_plane - ndc_z * (far_plane - near_plane);
    return (near_plane * far_plane) / (denom < 1e-6f ? 1e-6f : denom);
}

#ifdef __cplusplus
static_assert(std::is_standard_layout_v<ShadowInfo> && std::is_trivially_copyable_v<ShadowInfo>);
static_assert(sizeof(ShadowInfo) == 144);
static_assert(offsetof(ShadowInfo, light_vp) == 0);
static_assert(offsetof(ShadowInfo, texel_size) == 64);
static_assert(offsetof(ShadowInfo, normal_bias) == 68);
static_assert(offsetof(ShadowInfo, has_shadow) == 72);
static_assert(offsetof(ShadowInfo, layer) == 76);
static_assert(offsetof(ShadowInfo, light_near) == 80);
static_assert(offsetof(ShadowInfo, light_far) == 84);
static_assert(offsetof(ShadowInfo, light_size_uv) == 88);
static_assert(offsetof(ShadowInfo, projection_type) == 92);
static_assert(offsetof(ShadowInfo, light_position) == 96);
static_assert(offsetof(ShadowInfo, light_u) == 112);
static_assert(offsetof(ShadowInfo, light_v) == 128);

static_assert(std::is_standard_layout_v<ShadowVisibilityUniforms> &&
              std::is_trivially_copyable_v<ShadowVisibilityUniforms>);
static_assert(sizeof(ShadowVisibilityUniforms) == 80);
static_assert(offsetof(ShadowVisibilityUniforms, inv_view_proj) == 0);
static_assert(offsetof(ShadowVisibilityUniforms, viewport_size) == 64);
static_assert(offsetof(ShadowVisibilityUniforms, shadow_light_index) == 72);
static_assert(offsetof(ShadowVisibilityUniforms, frame_index) == 76);
}  // namespace pts::rendering
#endif

#undef PTS_SHARED
#undef PTS_SHARED_FUNCTION
#undef PTS_DEFAULT
