#pragma once
#include "stringManip.h"
#include "camera.h"
#include "lightData.h"
#include "object.h"
#include "params.h"
#include "shaderCommon.h"

namespace PTS {
	namespace VulkanRayTracingShaders {
		namespace _private {
			auto constexpr k_common_src = R"(
#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

struct Payload {
    // output from closest hit shader
    vec3 Li;  // sampled radiance
    vec3 pos; // intersection pos, world space
    vec3 wi;  // incoming dir, world space
    vec3 n;   // normal, world space
    float pdf; // pdf of wi
    bool done;

    // input to closest hit shader
    int iteration;
    int level;
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

// Hash Functions for GPU Rendering, Jarzynski et al.
// http://www.jcgt.org/published/0009/03/02/
vec3 random_pcg3d(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
  v ^= v >> 16u;
  v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
  return vec3(v) * (1.0/float(0xffffffffu));
}

mat3 normalToWorldSpace(in vec3 normal) {
   vec3 someVec = vec3(1.0, 0.0, 0.0);
   float dd = dot(someVec, normal);
   vec3 tangent = vec3(0.0, 1.0, 0.0);
   if(1.0 - abs(dd) > 1e-6) {
     tangent = normalize(cross(someVec, normal));
   }
   vec3 bitangent = cross(normal, tangent);
   return mat3(tangent, bitangent, normal);
}

mat3 worldToNormalSpace(in vec3 normal) {
    return transpose(normalToWorldSpace(normal));
}

vec3 Faceforward(vec3 n, vec3 v) {
    return (dot(n, v) < 0.f) ? -n : n;
}
bool SameHemisphere(vec3 w, vec3 wp) {
    return w.z * wp.z > 0;
}
vec3 squareToHemisphereCosine(vec2 xi) {
    float phi = TWO_PI * xi.x;
    float cosTheta = sqrt(xi.y);
    float sinTheta = sqrt(1.0 - xi.y);
    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}
float squareToHemisphereCosinePDF(vec3 sp) {
    return sp.z * INV_PI;
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
        payload = Payload(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), 0.0, false,
            perFrameData.iteration, i);

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
            // TODO: remove this hack
            color *= payload.Li * 2.;
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
    
    if (perFrameData.iteration > 0) {
        vec4 prevColor = imageLoad(outputImage, ivec2(gl_LaunchIDEXT.xy));
        prevColor.rgb = pow(prevColor.rgb, vec3(2.2));
        color = mix(prevColor.rgb, color, 1.0 / float(perFrameData.iteration));
    }
    color = pow(color, vec3(1.0 / 2.2));
    imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(color, 1.0));
}
)"sv;

			auto constexpr _k_miss_shader_src_glsl = R"(
layout(location = 0) rayPayloadInEXT Payload payload;

void main() {
    payload.Li = vec3(0.0);
    payload.done = true;
}
)"sv;

			// https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_ray_tracing.txt
			auto constexpr _k_closest_hit_shader_src_glsl = R"(
layout(location = 0) rayPayloadInEXT Payload payload;
hitAttributeEXT vec3 attribs;

// these functions operate in normal space
#define SAMPLE_F(name) vec3 name(in vec3 wo, in MaterialData material, out vec3 wi, out float pdf)

SAMPLE_F(diffuse) {
    vec2 xi = random_pcg3d(uvec3(gl_LaunchIDEXT.xy, payload.iteration + payload.level)).xy;
    wi = squareToHemisphereCosine(xi);
    pdf = squareToHemisphereCosinePDF(wi);
    return material.base_color * INV_PI;
}

SAMPLE_F(specular) {
    wi = vec3(wo.x, wo.y, -wo.z);
    pdf = 1.0;

    float absCosTheta = abs(wi.z);
    if (absCosTheta <= Epsilon) {
        // total internal reflection
        return vec3(0.0);
    }
    return material.base_color / absCosTheta;
}

void main() {
    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    // gl_InstanceCustomIndexEXT is unique ID for the mesh instance
    // gl_PrimitiveID is the index of the triangle in the mesh

    uvec3 triIndices = faceIndices[nonuniformEXT(gl_InstanceCustomIndexEXT)].data[gl_PrimitiveID];
    VertexAttribData vas[3];
    vas[0] = vertexAttribs[nonuniformEXT(gl_InstanceCustomIndexEXT)].data[int(triIndices.x)];
    vas[1] = vertexAttribs[nonuniformEXT(gl_InstanceCustomIndexEXT)].data[int(triIndices.y)];
    vas[2] = vertexAttribs[nonuniformEXT(gl_InstanceCustomIndexEXT)].data[int(triIndices.z)];
    vec3 n = baryLerp(vas[0].normal, vas[1].normal, vas[2].normal, bary);

    // gl_ObjectToWorldEXT is a 4x3 matrix (4 columns, 3 rows)    
    vec3 posW = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    mat3 invTrans = transpose(inverse(mat3(gl_ObjectToWorldEXT)));
    vec3 nW = normalize(invTrans * n);

    MaterialData material = materials[nonuniformEXT(gl_InstanceCustomIndexEXT)];
    if (material.emissive_color != vec3(0.0)) {
        payload.Li = material.emissive_color;
        payload.done = true;
    } else {
        vec3 wi; float pdf;
        vec3 Li = diffuse(worldToNormalSpace(nW) * (-gl_WorldRayDirectionEXT), material, wi, pdf);
        vec3 wiW = normalToWorldSpace(nW) * wi;
        payload.Li = Li;
        payload.pos = posW;
        payload.wi = wiW;
        payload.n = nW;
        payload.pdf = pdf;
        payload.done = false;
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
			_private::k_per_frame_data_decl,
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
			LightData::glsl_def,
			_private::k_vertex_attribs_decl,
			_private::k_indices_decl,
			_private::k_material_uniform_decl,
			_private::k_light_uniform_decl,
			_private::_k_closest_hit_shader_src_glsl
		>;
	} // namespace VulkanRayTracingShaders
} // namespace PTS
