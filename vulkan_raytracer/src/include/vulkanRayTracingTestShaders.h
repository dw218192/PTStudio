#pragma once
#include "shaderCommon.h"

namespace PTS {
	namespace VulkanRayTracingShaders {
		namespace _private {
			auto constexpr _k_test_common_src = R"(
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
struct Payload {
  vec3 color;
  bool shadowRayMiss;
}; // type of the "payload" variable

const float PI = 3.1415926535897;

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

			auto constexpr _k_test_raygen_src = R"(
// intersection data
layout(location = 0) rayPayloadEXT Payload payload;

void main() {
    vec2 uv = vec2(gl_LaunchIDEXT.xy) / vec2(gl_LaunchSizeEXT.xy); // uv in [0, 1]
    vec2 ndc = uv * 2.0 - 1.0; // uv in [-1, 1]
    vec4 uv_world = perFrameData.camera.inv_view_proj * vec4(ndc, 1.0, 1.0);
    vec3 rd = normalize(uv_world.xyz / uv_world.w - perFrameData.camera.pos); // ray direction in world space
    vec3 ro = perFrameData.camera.pos; // ray origin in world space

    traceRayEXT(
        topLevelAS, // acceleration structure
        gl_RayFlagsNoneEXT, // ray flags
        0xFF, // cull mask
        0, // sbt record offset
        0, // sbt record stride
        0, // miss index
        ro, // ray origin
        0.0, // ray tmin
        rd, // ray direction
        100000.0, // ray tmax
        0 // payload location
    );

    imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(payload.color, 1.0));
}
)"sv;

			auto constexpr _k_test_miss_src = R"(
layout(location = 0) rayPayloadInEXT Payload payload;

void main() {
    payload.color = vec3(0.0);
    payload.shadowRayMiss = true;
}
)"sv;

			auto constexpr _k_test_closest_hit_src = R"(
layout(location = 0) rayPayloadInEXT Payload payload;
hitAttributeEXT vec3 attribs;

void main() {
    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    uvec3 triIndices = faceIndices[nonuniformEXT(gl_InstanceCustomIndexEXT)].data[gl_PrimitiveID];
    VertexAttribData vas[3];
    vas[0] = vertexAttribs[nonuniformEXT(gl_InstanceCustomIndexEXT)].data[int(triIndices.x)];
    vas[1] = vertexAttribs[nonuniformEXT(gl_InstanceCustomIndexEXT)].data[int(triIndices.y)];
    vas[2] = vertexAttribs[nonuniformEXT(gl_InstanceCustomIndexEXT)].data[int(triIndices.z)];
    vec3 n = baryLerp(vas[0].normal, vas[1].normal, vas[2].normal, bary);

    vec3 position = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    mat3 inner = mat3(gl_ObjectToWorldEXT[0].xyz, gl_ObjectToWorldEXT[1].xyz, gl_ObjectToWorldEXT[2].xyz);
    mat3 invTrans = transpose(inverse(inner));
    vec3 normal = normalize(invTrans * n);

    // prepare shadow ray
    vec3 lightPos = vec3(0.0, 3.0, 0.0);
    vec3 lightDir = normalize(lightPos - position);

    uint rayFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;
    float rayMin     = 0.001;
    float rayMax     = length(lightPos - position);  
    float shadowBias = 0.001;
    uint cullMask = 0xFFu;
    float frontFacing = dot(-gl_WorldRayDirectionEXT, normal);
    vec3 shadowRayOrigin = position + sign(frontFacing) * shadowBias * normal;
    vec3 shadowRayDirection = lightDir;
    payload.shadowRayMiss = false;

    // shot shadow ray
    traceRayEXT(topLevelAS, rayFlags, cullMask, 0u, 0u, 0u, 
        shadowRayOrigin, rayMin, shadowRayDirection, rayMax, 0);

    const vec3 ambientColor = vec3(0.2, 0.2, 0.2);
    const vec3 baseColor = vec3(0.7, 0.7, 0.7);
    
    // diffuse shading
    vec3 radiance = ambientColor; // ambient term
    if(payload.shadowRayMiss) { // if not in shadow
        float irradiance = max(dot(lightDir, normal), 0.0);
        if(irradiance > 0.0) { // if receives light
            radiance += baseColor * irradiance; // diffuse shading
        }
    }
    payload.color = radiance;
}  
)"sv;
		} // namespace _private

		// produce the final shader source
		auto constexpr k_test_ray_gen_shader_src_glsl = PTS::join_v<
			_private::_k_test_common_src,
			CameraData::k_glsl_decl,
			PerFrameData::k_glsl_decl,
			_private::k_accel_struct_decl,
			_private::k_per_frame_data_decl,
			_private::k_output_image_decl,
			_private::_k_test_raygen_src
		>;

		auto constexpr k_test_miss_shader_src_glsl = PTS::join_v<
			_private::_k_test_common_src,
			_private::_k_test_miss_src
		>;

		auto constexpr k_test_closest_hit_shader_src_glsl = PTS::join_v<
			_private::_k_test_common_src,
			MaterialData::k_glsl_decl,
			VertexAttribData::k_glsl_decl,
			_private::k_vertex_attribs_decl,
			_private::k_indices_decl,
			_private::k_accel_struct_decl,
			_private::_k_test_closest_hit_src
		>;
	} // namespace VulkanRayTracingShaders
} // namespace PTS
