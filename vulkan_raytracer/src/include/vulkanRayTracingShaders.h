#pragma once
#include "stringManip.h"
#include "camera.h"
#include "object.h"
#include "params.h"

namespace PTS {

struct CameraData {
    glm::mat4 inv_view_proj; // 0
    glm::vec3 cam_pos;       // 64
    unsigned char _pad[4] = { 0, 0, 0, 0 };
    // total size: 80

    CameraData() = default;
    explicit CameraData(Camera const& camera) : 
        inv_view_proj{camera.get_inv_view_proj()}, cam_pos{camera.get_eye()} {}

    auto operator==(CameraData const& other) const noexcept -> bool {
        return inv_view_proj == other.inv_view_proj
            && cam_pos == other.cam_pos;
    }
    auto operator!=(CameraData const& other) const noexcept -> bool {
        return !(*this == other);
    }

    static constexpr auto k_glsl_decl = std::string_view { "\
        struct CameraData {\n\
            mat4 inv_view_proj;\n\
            vec3 pos;\n\
        };\n" };
};
static_assert(sizeof(CameraData) == 80, "CameraData size mismatch");

struct PerFrameData {
    CameraData camera; // 0
    int iteration;     // 80
    int num_samples;   // 84
    int max_bounces;   // 88
    // total size: 96

    static constexpr auto k_glsl_decl = std::string_view { "\
        struct PerFrameData {\n\
            CameraData camera;\n\
            int iteration;\n\
            int num_samples;\n\
            int max_bounces;\n\
        };\n" };
};

struct VertexData {
    glm::vec3 position;

    VertexData() = default;
    explicit VertexData(Vertex const& vertex) : position{vertex.position} {}
};
struct MaterialData {
    glm::vec3 base_color;                          // 0    base alignment: 16
    unsigned char _pad1[4] = { 0, 0, 0, 0 };       // 12
    glm::vec3 emissive_color;                      // 16   base alignment: 16
    unsigned char _pad2[4] = { 0, 0, 0, 0 };       // 28
    // total size: 32

    MaterialData() = default;
    explicit MaterialData(Material const& material) : 
        base_color{material.albedo}, emissive_color{material.emission} {}

    auto operator==(MaterialData const& other) const noexcept -> bool {
        return base_color == other.base_color
            && emissive_color == other.emissive_color;
    }
    auto operator!=(MaterialData const& other) const noexcept -> bool {
        return !(*this == other);
    }

    static constexpr auto k_glsl_decl = std::string_view { "\
        struct MaterialData {\n\
            vec3 base_color;\n\
            vec3 emissive_color;\n\
        };\n" };
};
static_assert(sizeof(MaterialData) == 32, "MaterialData size mismatch");

struct VertexAttribData {
    glm::vec3 normal;     // 0   base alignment: 16
    unsigned char _pad1[4] = { 0, 0, 0, 0 };
    glm::vec2 tex_coord;  // 16  base alignment: 8
    unsigned char _pad2[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    VertexAttribData() = default;
    explicit VertexAttribData(Vertex const& vertex) : 
        normal{vertex.normal}, tex_coord{vertex.uv} {}

    auto operator==(VertexAttribData const& other) const noexcept -> bool {
        return normal == other.normal
            && tex_coord == other.tex_coord;
    }
    auto operator!=(VertexAttribData const& other) const noexcept -> bool {
        return !(*this == other);
    }

    static constexpr auto k_glsl_decl = std::string_view { "\
        struct VertexAttribData {\n\
            vec3 normal;\n\
            vec2 tex_coord;\n\
        };\n" };
};
static_assert(sizeof(VertexAttribData) == 32, "VertexAttribData size mismatch");

struct FaceIndexData {
    glm::uvec3 indices;
    unsigned char _pad[4] = { 0, 0, 0, 0 };

    FaceIndexData() = default;
    explicit FaceIndexData(unsigned i0, unsigned i1, unsigned i2) : 
        indices{i0, i1, i2} {} 
};
static_assert(sizeof(FaceIndexData) == 16, "FaceIndexData size mismatch");

struct Binding {
    int set, binding;
};
struct RayTracingBindings {
    static constexpr Binding ACCEL_STRUCT_BINDING { 0, 0 };
    static constexpr Binding MATERIALS_BINDING { 0, 1 };
    static constexpr Binding OUTPUT_IMAGE_BINDING { 0, 2 };

    static constexpr Binding VERTEX_ATTRIBS_BINDING { 1, 0 };
    static constexpr Binding INDICES_BINDING { 2, 0 };

    static constexpr int k_num_bindings = 5;
    static constexpr int k_num_sets = 3;
};

namespace _private {
// silence MSVC warning C4455 which is a bug
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4455)
using std::literals::string_view_literals::operator""sv;
#pragma warning(pop)
#endif

// meta function to generate glsl declaration
template <int set, int binding, std::string_view const& decl, std::string_view const&... mods>
struct gen_glsl_decl {
    static auto constexpr _0 = "layout (set ="sv;
    static auto constexpr _1 = ", binding ="sv;
    static auto constexpr _2 = ") "sv;
    static auto constexpr _3 = ", "sv;
    static auto constexpr impl() {
        if constexpr (sizeof...(mods) == 0) {
            return PTS::join_v<_0, PTS::to_str_v<set>, _1, PTS::to_str_v<binding>, _2, decl>;
        } else {
            return PTS::join_v<_0, PTS::to_str_v<set>, _1, PTS::to_str_v<binding>, _3, PTS::join_v<mods...>, _2, decl>;
        }
    }
    static auto constexpr value = impl();
};

auto constexpr k_common_src = R"(
#version 460

#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_tracing_position_fetch : enable
#extension GL_EXT_nonuniform_qualifier : enable

struct Payload {
    vec3 Li;  // sampled radiance
    vec3 pos; // intersection pos
    vec3 wi;  // incoming dir
    vec3 n;   // normal
    float pdf; // pdf of wi
    bool done;
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
float rng(uvec2 seed) {
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
vec3 squareToDiskConcentric(vec2 xi) {
    float theta, r;
    float a = 2 * xi.x - 1;
    float b = 2 * xi.y - 1;
    float c = PI * 0.25f;
    if(a > -b) {
        if(a > b) {
            r = a;
            theta = c * b / a;
        } else {
            r = b;
            theta = c * (2 - a / b);
        }
    } else {
        if(a < b) {
            r = -a;
            theta = c * (4 + b / a);
        } else {
            r = -b;
            if(b != 0) {
                theta = c * (6 - a / b);
            } else {
                theta = 0;
            }
        }
    }
    return vec3(r * cos(theta), r * sin(theta), 0);
}
vec3 squareToHemisphereCosine(vec2 xi) {
    vec3 ret = squareToDiskConcentric(xi);
    ret.z = sqrt(max(0.f, 1 - ret.x * ret.x - ret.y * ret.y));
    return ret;
}
float squareToHemisphereCosinePDF(vec3 sp) {
    return sp.z * INV_PI;
}
vec3 squareToSphereUniform(vec2 sp) {
    float z = 1 - 2 * sp.x;
    float b = sqrt(1 - z * z);
    float phi = 2 * PI * sp.y;

    return vec3 (
        cos(phi) * b,
        sin(phi) * b,
        z
    );
}
float squareToSphereUniformPDF(vec3 sp) {
    return INV_FOUR_PI;
}

vec2 baryLerp(vec2 a, vec2 b, vec2 c, vec3 bary) {
    return vec2(
        bary.x * a.x + bary.y * b.x + bary.z * c.x,
        bary.x * a.y + bary.y * b.y + bary.z * c.y
    );
}

vec3 baryLerp(vec3 a, vec3 b, vec3 c, vec3 bary) {
    return vec3(
        bary.x * a.x + bary.y * b.x + bary.z * c.x,
        bary.x * a.y + bary.y * b.y + bary.z * c.y,
        bary.x * a.z + bary.y * b.z + bary.z * c.z
    );
}
)"sv;

auto constexpr _k_vertex_attribs_decl = R"(
readonly buffer VertexAttribsBlock {
    VertexAttribData data[];
} vertexAttribs[];
)"sv;
auto constexpr k_vertex_attribs_decl = gen_glsl_decl<
    RayTracingBindings::VERTEX_ATTRIBS_BINDING.set,
    RayTracingBindings::VERTEX_ATTRIBS_BINDING.binding,
    _k_vertex_attribs_decl
>::value;

// indices declaration
auto constexpr _k_indices_decl = R"(
readonly buffer FaceIndicesBlock {
    uvec3 data[];
} faceIndices[];
)"sv;
auto constexpr k_indices_decl = gen_glsl_decl<
    RayTracingBindings::INDICES_BINDING.set,
    RayTracingBindings::INDICES_BINDING.binding,
    _k_indices_decl
>::value;

// material uniform declaration
auto constexpr _k_material_uniform_decl_0 = R"(
uniform MaterialBlock {
    MaterialData materials[)"sv;
auto constexpr _k_material_uniform_decl_1 = R"(];
};
)"sv;
auto constexpr _k_material_uniform_decl = PTS::join_v<
    _k_material_uniform_decl_0,
    PTS::to_str_v<k_max_instances>,
    _k_material_uniform_decl_1
>;
auto constexpr k_material_uniform_decl = gen_glsl_decl<
    RayTracingBindings::MATERIALS_BINDING.set,
    RayTracingBindings::MATERIALS_BINDING.binding,
    _k_material_uniform_decl
>::value;

// camera uniform declaration
auto constexpr k_camera_uniform_decl = R"(
layout (push_constant) uniform PerFrameDataBlock {
    PerFrameData perFrameData;
};
)"sv;

// output image declaration
auto constexpr _k_output_image_decl = R"(
uniform image2D outputImage;
)"sv;
auto constexpr _k_output_image_mod = "rgba8"sv;
auto constexpr k_output_image_decl = gen_glsl_decl<
    RayTracingBindings::OUTPUT_IMAGE_BINDING.set,
    RayTracingBindings::OUTPUT_IMAGE_BINDING.binding,
    _k_output_image_decl,
    _k_output_image_mod
>::value;

// acceleration structure declaration
auto constexpr _k_accel_struct_decl = R"(
uniform accelerationStructureEXT topLevelAS;
)"sv;

auto constexpr k_accel_struct_decl = gen_glsl_decl<
    RayTracingBindings::ACCEL_STRUCT_BINDING.set,
    RayTracingBindings::ACCEL_STRUCT_BINDING.binding,
    _k_accel_struct_decl
>::value;

auto constexpr _k_ray_gen_shader_src_glsl = R"(
// intersection data
layout(location = 0) rayPayloadEXT Payload payload;

void main() {
    vec2 uv = vec2(gl_LaunchIDEXT.xy) / vec2(gl_LaunchSizeEXT.xy); // uv in [0, 1]
    vec2 ndc = uv * 2.0 - 1.0; // uv in [-1, 1]
    vec4 uv_world = perFrameData.camera.inv_view_proj * vec4(ndc, 1.0, 1.0);
    vec3 rd = normalize(uv_world.xyz / uv_world.w - perFrameData.camera.pos); // ray direction in world space
    vec3 ro = perFrameData.camera.pos; // ray origin in world space

    vec3 color = vec3(1.0);
    for (int i = 0; i < perFrameData.max_bounces; ++i) {
        payload = Payload(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), 0.0, false);
        traceRayEXT(
            topLevelAS,
            gl_RayFlagsOpaqueEXT, // target opaque geometry
            0xFF,  // cull mask
            0, // sbt record offset
            0, // sbt record stride
            0, // miss index
            ro, // ray origin
            0.001, // ray tmin
            rd, // ray direction
            100000.0, // ray tmax
            0 // payload location
        );
        if (payload.done) {
            color *= payload.Li;
            break;
        } else {
            if (payload.pdf <= Epsilon || payload.Li == vec3(0.0) || payload.wi == vec3(0.0)) {
                // invalid path
                color = vec3(0.0);
                break;
            }
            color *= payload.Li * abs(dot(payload.wi, payload.n)) / payload.pdf;
            ro = payload.pos;
            rd = payload.wi;
        }
    }

    if (!payload.done) {
        // didn't hit any light source
        color = vec3(0.0);
    }
    
    vec4 prevColor = imageLoad(outputImage, ivec2(gl_LaunchIDEXT.xy));
    color = mix(prevColor.rgb, color, 1.0 / float(perFrameData.iteration));
    imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(color, 1.0));
}
)"sv;

auto constexpr _k_miss_shader_src_glsl = R"(
layout(location = 0) rayPayloadInEXT Payload payload;

void main() {
    payload = Payload(vec3(0.0, 0.0, 0.0), vec3(0.0), vec3(0.0), vec3(0.0), 0.0, true);
}
)"sv;

// https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_ray_tracing.txt
auto constexpr _k_closest_hit_shader_src_glsl = R"(
layout(location = 0) rayPayloadInEXT Payload payload;
hitAttributeEXT vec3 attribs;

void main() {
    vec3 ps[3];
    ps[0] = gl_HitTriangleVertexPositionsEXT[0];
    ps[1] = gl_HitTriangleVertexPositionsEXT[1];
    ps[2] = gl_HitTriangleVertexPositionsEXT[2];

    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 pos = baryLerp(ps[0], ps[1], ps[2], bary);

    // gl_InstanceCustomIndexEXT is unique ID for the mesh instance
    // gl_PrimitiveID is the index of the triangle in the mesh

    uvec3 triIndices = faceIndices[nonuniformEXT(gl_InstanceCustomIndexEXT)].data[gl_PrimitiveID];
    VertexAttribData vas[3];
    vas[0] = vertexAttribs[nonuniformEXT(gl_InstanceCustomIndexEXT)].data[int(triIndices.x)];
    vas[1] = vertexAttribs[nonuniformEXT(gl_InstanceCustomIndexEXT)].data[int(triIndices.y)];
    vas[2] = vertexAttribs[nonuniformEXT(gl_InstanceCustomIndexEXT)].data[int(triIndices.z)];
    vec3 n = baryLerp(vas[0].normal, vas[1].normal, vas[2].normal, bary);

    // gl_ObjectToWorldEXT is a 4x3 matrix (4 columns, 3 rows)
    vec3 posW = gl_ObjectToWorldEXT * vec4(pos, 1.0);
    mat3 inner = mat3(gl_ObjectToWorldEXT[0].xyz, gl_ObjectToWorldEXT[1].xyz, gl_ObjectToWorldEXT[2].xyz);
    mat3 invTrans = transpose(inverse(inner));
    vec3 nW = normalize(invTrans * n);

    MaterialData material = materials[gl_InstanceCustomIndexEXT];
    if (material.emissive_color != vec3(0.0)) {
        payload = Payload(material.emissive_color, posW, vec3(0.0), nW, 0.0, true);
    } else {
        // diffuse
        uvec2 seed = uvec2(gl_LaunchIDEXT.xy);
        vec2 xi = vec2(rng(seed), rng(seed));
        vec3 wi = squareToHemisphereCosine(xi);
        float pdf = squareToHemisphereCosinePDF(wi);
        vec3 wiW = LocalToWorld(nW) * wi;
        vec3 Li = material.base_color * INV_PI;
        payload = Payload(Li, posW, wiW, nW, pdf, false);
    }
}
)"sv;
} // namespace _private

// produce the final shader source
auto constexpr k_ray_gen_shader_src_glsl = PTS::join_v<
    _private::k_common_src,
    CameraData::k_glsl_decl,
    PerFrameData::k_glsl_decl,
    MaterialData::k_glsl_decl,
    VertexAttribData::k_glsl_decl,
    _private::k_vertex_attribs_decl,
    _private::k_indices_decl,
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
    MaterialData::k_glsl_decl,
    VertexAttribData::k_glsl_decl,
    _private::k_vertex_attribs_decl,
    _private::k_indices_decl,
    _private::k_material_uniform_decl,
    _private::_k_closest_hit_shader_src_glsl
>;

} // namespace PTS