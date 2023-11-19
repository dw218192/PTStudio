#pragma once
#include "concat.h"

using std::literals::string_view_literals::operator""sv;

auto constexpr k_ray_gen_shader_test_glsl = R"(
#version 460
#extension GL_EXT_ray_tracing : enable
// per-scene data
layout (set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout (set = 0, binding = 1, rgba8) uniform image2D outputImage;
void main() {
    vec2 uv = vec2(gl_LaunchIDEXT.xy) / vec2(gl_LaunchSizeEXT.xy); // uv in [0, 1]
    imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(uv, 0.0, 1.0));
}
)"sv;


auto constexpr k_common_src = R"(
#version 460
#extension GL_EXT_ray_tracing : enable
struct Payload {
    vec3 color;
};
)"sv;

auto constexpr _k_ray_gen_shader_src_glsl = R"(
// per-scene data
layout (set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout (set = 0, binding = 1, rgba8) uniform image2D outputImage;
// per-frame data
layout (set = 0, binding = 2) uniform CameraData {
    mat4 inv_view_proj;
    vec3 cam_pos;
};

// intersection data
layout(location = 0) rayPayloadEXT Payload payload;

void main() {
    vec2 uv = vec2(gl_LaunchIDEXT.xy) / vec2(gl_LaunchSizeEXT.xy); // uv in [0, 1]
    vec2 ndc = uv * 2.0 - 1.0; // uv in [-1, 1]
    vec4 uv_world = inv_view_proj * vec4(ndc, 1.0, 1.0);
    vec3 direction = normalize(uv_world.xyz / uv_world.w - cam_pos); // ray direction in world space

    traceRayEXT(
        topLevelAS,
        gl_RayFlagsOpaqueEXT, // target opaque geometry
        0xFF,  // cull mask
        0, // sbt record offset
        0, // sbt record stride
        0, // miss index
        cam_pos, // ray origin
        0.001, // ray tmin
        direction, // ray direction
        100000.0, // ray tmax
        0 // payload location
    );

    imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(payload.color, 1.0));
}
)"sv;

auto constexpr _k_miss_shader_src_glsl = R"(
layout(location = 0) rayPayloadInEXT Payload payload;

void main() {
    payload.color = vec3(0.0);
}
)"sv;

auto constexpr _k_closest_hit_shader_src_glsl = R"(
layout(location = 0) rayPayloadInEXT Payload payload;

void main() {
    payload.color = vec3(1.0);
}
)"sv;

auto constexpr k_ray_gen_shader_src_glsl = PTS::join_v<
    k_common_src,
    _k_ray_gen_shader_src_glsl
>;

auto constexpr k_miss_shader_src_glsl = PTS::join_v<
    k_common_src,
    _k_miss_shader_src_glsl
>;

auto constexpr k_closest_hit_shader_src_glsl = PTS::join_v<
    k_common_src,
    _k_closest_hit_shader_src_glsl
>;