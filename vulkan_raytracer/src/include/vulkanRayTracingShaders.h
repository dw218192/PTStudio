#pragma once
#include "stringManip.h"
#include "camera.h"
#include "object.h"

using std::literals::string_view_literals::operator""sv;

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

    MaterialData() = default;
    MaterialData(Material const& material) : base_color{material.albedo} {}
};

[[nodiscard]] inline auto to_rt_data(Camera const& camera) -> CameraData {
    return {
        camera.get_inv_view_proj(),
        camera.get_eye()
    };
}
[[nodiscard]] inline auto to_rt_data(Material const& material) -> MaterialData {
    return { material};
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

auto constexpr k_common_src = R"(
#version 460
#extension GL_EXT_ray_tracing : enable
struct Payload {
    vec3 color;
};
struct Material {
    vec3 base_color;
};
struct CameraData
{
    mat4 inv_view_proj;
    vec3 pos;
};
)"sv;

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
}
)"sv;

auto constexpr _k_closest_hit_shader_src_glsl = R"(
layout(location = 0) rayPayloadInEXT Payload payload;

void main() {
    payload.color = materials[gl_InstanceCustomIndexEXT].base_color;
}
)"sv;


// produce the final shader source
auto constexpr k_ray_gen_shader_src_glsl = PTS::join_v<
    k_common_src,
    k_accel_struct_decl,
    k_camera_uniform_decl,
    k_material_uniform_decl,
    k_output_image_decl,
    _k_ray_gen_shader_src_glsl
>;

auto constexpr k_miss_shader_src_glsl = PTS::join_v<
    k_common_src,
    _k_miss_shader_src_glsl
>;

auto constexpr k_closest_hit_shader_src_glsl = PTS::join_v<
    k_common_src,
    k_material_uniform_decl,
    _k_closest_hit_shader_src_glsl
>;

} // namespace PTS