#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

#include "common.inc"
#include "bindings.inc"

#include "mesh_attrs.inc"

#include "perframe_data.inc"

layout(set = TOP_LEVEL_ACC_SET, binding = TOP_LEVEL_ACC_BINDING)
uniform accelerationStructureEXT topLevelAS;

layout(set = OUTPUT_IMG_SET, binding = OUTPUT_IMG_BINDING, rgba8) 
uniform image2D outputImage;

// intersection data
layout(location = 0)
rayPayloadEXT Payload payload;

void main() {
    vec2 uv = vec2(gl_LaunchIDEXT.xy) / vec2(gl_LaunchSizeEXT.xy); // uv in [0, 1]
    vec2 ndc = uv * 2.0 - 1.0; // uv in [-1, 1]
    vec4 uv_world = perFrameData.camera.inv_view_proj * vec4(ndc, 1.0, 1.0);
    vec3 rd = normalize(uv_world.xyz / uv_world.w - perFrameData.camera.pos); // ray direction in world space
    vec3 ro = perFrameData.camera.pos; // ray origin in world space

    vec3 color = vec3(1.0);
    for (int i = 0; i < perFrameData.max_bounces; ++i) {
        payload = Payload(vec3(0.0), vec3(0.0), vec3(0.0), false, false,
            perFrameData.iteration,
            i,
            perFrameData.direct_lighting_only);

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
            if (!payload.invalid && !payload.done) {
                color *= payload.Li;
                ro = payload.pos;
                rd = payload.wi;
            } else {
                // invalid sample
                color = vec3(0.0);
                break;
            }
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