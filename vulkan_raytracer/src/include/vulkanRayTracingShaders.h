#pragma once
#include "stringManip.h"
#include "camera.h"
#include "object.h"

namespace PTS {

struct CameraData {
    glm::mat4 inv_view_proj;
    glm::vec3 cam_pos;
};
struct VertexData {
    glm::vec3 position;

    VertexData() = default;
    VertexData(Vertex const& vertex) : position{vertex.position} {}
};
struct MaterialData {
    glm::vec3 base_color;
    glm::vec3 emissive_color;

    MaterialData() = default;
    MaterialData(Material const& material) : 
        base_color{material.albedo}, emissive_color{material.emission} {}
};

[[nodiscard]] inline auto to_rt_data(Camera const& camera) -> CameraData {
    return {
        camera.get_inv_view_proj(),
        camera.get_eye()
    };
}
[[nodiscard]] inline auto to_rt_data(Material const& material) -> MaterialData {
    return { material };
}
[[nodiscard]] inline auto to_rt_data(tcb::span<Vertex const> vertices) -> std::vector<VertexData> {
    std::vector<VertexData> result;
    result.reserve(vertices.size());
    for (auto const& vertex : vertices) {
        result.emplace_back(vertex);
    }
    return result;
}

enum RTBindings {
    ACCEL_STRUCT_BINDING = 0,
    CAMERA_BINDING = 1,
    MATERIALS_BINDING = 2,
    OUTPUT_IMAGE_BINDING = 3,
};

namespace _private {
using std::literals::string_view_literals::operator""sv;

auto constexpr k_common_src = R"(
#version 460

#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_tracing_position_fetch : enable

struct Payload {
    vec3 brdf;
    vec3 color;
    vec3 position;
    vec3 normal;
    bool done;
};
struct Material {
    vec3 base_color;
    vec3 emissive_color;
};
struct CameraData {
    mat4 inv_view_proj;
    vec3 pos;
};

// utility functions and constants
#define PI               3.14159265358979323
#define TWO_PI           6.28318530717958648
#define FOUR_PI          12.5663706143591729
#define INV_PI           0.31830988618379067
#define INV_TWO_PI       0.15915494309
#define INV_FOUR_PI      0.07957747154594767
#define PI_OVER_TWO      1.57079632679489662
#define ONE_THIRD        0.33333333333333333
#define E                2.71828182845904524
#define INFINITY         1000000.0
#define OneMinusEpsilon  0.99999994
#define RayEpsilon       0.000005
#define Epsilon          0.000001

// from ShaderToy https://www.shadertoy.com/view/4tXyWN
uvec2 seed;
float rng() {
    seed += uvec2(1);
    uvec2 q = 1103515245U * ( (seed >> 1U) ^ (seed.yx) );
    uint  n = 1103515245U * ( (q.x) ^ (q.y >> 3U) );
    return float(n) * (1.0 / float(0xffffffffU));
}

void coordinateSystem(in vec3 v1, out vec3 v2, out vec3 v3) {
    if (abs(v1.x) > abs(v1.y))
        v2 = vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        v2 = vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = cross(v1, v2);
}
mat3 LocalToWorld(vec3 nor) {
    vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return mat3(tan, bit, nor);
}
mat3 WorldToLocal(vec3 nor) {
    return transpose(LocalToWorld(nor));
}
vec3 Faceforward(vec3 n, vec3 v) {
    return (dot(n, v) < 0.f) ? -n : n;
}
bool SameHemisphere(vec3 w, vec3 wp) {
    return w.z * wp.z > 0;
}

)"sv;

// material uniform declaration
auto constexpr _k_material_uniform_decl_0 = R"(
layout (set = 0, binding = )"sv;
auto constexpr _k_material_uniform_decl_1 = R"() uniform MaterialBlock {
    Material materials[)"sv;
auto constexpr _k_material_uniform_decl_2 = R"(];
};
)"sv;
auto constexpr k_material_uniform_decl = PTS::join_v<
    _k_material_uniform_decl_0,
    PTS::to_str_v<RTBindings::MATERIALS_BINDING>,
    _k_material_uniform_decl_1,
    PTS::to_str_v<k_max_instances>,
    _k_material_uniform_decl_2
>;

// camera uniform declaration
auto constexpr _k_camera_uniform_decl_0 = R"(
layout (set = 0, binding = )"sv;
auto constexpr _k_camera_uniform_decl_1 = R"() uniform CameraBlock {
    CameraData camera;
};
)"sv;
auto constexpr k_camera_uniform_decl = PTS::join_v<
    _k_camera_uniform_decl_0,
    PTS::to_str_v<RTBindings::CAMERA_BINDING>,
    _k_camera_uniform_decl_1
>;

// output image declaration
auto constexpr _k_output_image_decl_0 = R"(
layout (set = 0, binding = )"sv;
auto constexpr _k_output_image_decl_1 = R"(, rgba8) uniform image2D outputImage;
)"sv;
auto constexpr k_output_image_decl = PTS::join_v<
    _k_output_image_decl_0,
    PTS::to_str_v<RTBindings::OUTPUT_IMAGE_BINDING>,
    _k_output_image_decl_1
>;

// acceleration structure declaration
auto constexpr _k_accel_struct_decl_0 = R"(
layout (set = 0, binding = )"sv;
auto constexpr _k_accel_struct_decl_1 = R"() uniform accelerationStructureEXT topLevelAS;
)"sv;
auto constexpr k_accel_struct_decl = PTS::join_v<
    _k_accel_struct_decl_0,
    PTS::to_str_v<RTBindings::ACCEL_STRUCT_BINDING>,
    _k_accel_struct_decl_1
>;

auto constexpr _k_ray_gen_shader_src_glsl = R"(
// intersection data
layout(location = 0) rayPayloadEXT Payload payload;

void main() {
    vec2 uv = vec2(gl_LaunchIDEXT.xy) / vec2(gl_LaunchSizeEXT.xy); // uv in [0, 1]
    vec2 ndc = uv * 2.0 - 1.0; // uv in [-1, 1]
    vec4 uv_world = camera.inv_view_proj * vec4(ndc, 1.0, 1.0);
    vec3 direction = normalize(uv_world.xyz / uv_world.w - camera.pos); // ray direction in world space

    traceRayEXT(
        topLevelAS,
        gl_RayFlagsOpaqueEXT, // target opaque geometry
        0xFF,  // cull mask
        0, // sbt record offset
        0, // sbt record stride
        0, // miss index
        camera.pos, // ray origin
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
    payload.done = true;
}
)"sv;

auto constexpr _k_closest_hit_shader_src_glsl = R"(
layout(location = 0) rayPayloadInEXT Payload payload;
hitAttributeEXT vec3 attribs;

// vec3 Li_Direct(vec3 p, vec3 n, vec3 wo, Material material) {
//     vec3 throughput = vec3(1.0);
// }
vec3 calcNormal(vec3 p0, vec3 p1, vec3 p2) {
    return -normalize(cross(p1 - p0, p2 - p0));
}

void main() {
    vec3 ps[3];
    ps[0] = gl_HitTriangleVertexPositionsEXT[0];
    ps[1] = gl_HitTriangleVertexPositionsEXT[1];
    ps[2] = gl_HitTriangleVertexPositionsEXT[2];

    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 pos = ps[0] * bary.x + ps[1] * bary.y + ps[2] * bary.z;
    vec3 n = calcNormal(ps[0], ps[1], ps[2]);

    Material material = materials[gl_InstanceCustomIndexEXT];
    
    payload.color = material.base_color;
    payload.done = true;
}
)"sv;
} // namespace _private

// produce the final shader source
auto constexpr k_ray_gen_shader_src_glsl = PTS::join_v<
    _private::k_common_src,
    _private::k_accel_struct_decl,
    _private::k_camera_uniform_decl,
    _private::k_material_uniform_decl,
    _private::k_output_image_decl,
    _private::_k_ray_gen_shader_src_glsl
>;

auto constexpr k_miss_shader_src_glsl = PTS::join_v<
    _private::k_common_src,
    _private::_k_miss_shader_src_glsl
>;

auto constexpr k_closest_hit_shader_src_glsl = PTS::join_v<
    _private::k_common_src,
    _private::k_material_uniform_decl,
    _private::_k_closest_hit_shader_src_glsl
>;

} // namespace PTS