#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

#include "common.inc"
#include "bindings.inc"

#include "mesh_attrs.inc"
#include "light.inc"

layout(set = TOP_LEVEL_ACC_SET, binding = TOP_LEVEL_ACC_BINDING)
uniform accelerationStructureEXT topLevelAS;

layout(location = 0)
rayPayloadInEXT Payload payload;

layout(location = 1)
rayPayloadEXT bool vis_test;

hitAttributeEXT vec3 attribs;

bool is_visible(in vec3 p, in vec3 q) {
    vec3 dir = q - p;
    float dist = length(dir);
    dir = normalize(dir);
    vec3 ro = p + dir * RayEpsilon;
    vec3 rd = dir;
    vis_test = true;
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        0xFF,  // cull mask
        0, // sbt record offset
        0, // sbt record stride
        0, // miss index
        ro, // ray origin
        0.001, // ray tmin
        rd, // ray direction
        dist + 0.01, // ray tmax
        1 // payload location
    );
    return !vis_test;
}

// area integration for light sampling
// see veach's thesis
float power_heuristic(float nf, float fPdf, float ng, float gPdf) {
    float f = nf * fPdf, g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

float pdf_light(in vec3 p, in vec3 n, in vec3 wiW, int light_idx) {
    if (light_idx < 0 || light_idx >= num_lights) {
        return 0.0;
    }

    LightData light = lights[light_idx];
    // point light
    if (light.type == POINT_LIGHT) {
        return 1.0 / float(num_lights);
    }
    return 0.0;
}
vec3 sample_light(in vec3 p, in vec3 n, out vec3 wiW, out float pdf, out int idx) {
    wiW = vec3(0.0);
    pdf = 0.0;
    if (num_lights == 0) {
        return vec3(0.0);
    }

    idx = int(random_pcg3d(uvec3(gl_LaunchIDEXT.xy, payload.iteration + payload.level)).x * float(num_lights));
    LightData light = lights[idx];
    vec3 light_pos = vec3(light.transform[3]);
    vec3 light_dir = light_pos - p;
    if (!is_visible(p, light_pos)) {
        return vec3(0.0);
    }

    pdf = pdf_light(p, n, wiW, idx);
    // point light
    if (light.type == POINT_LIGHT) {
        wiW = normalize(light_dir);
        return light.color * light.intensity * abs(dot(wiW, n))
            / (FOUR_PI * vec3(dot(light_dir, light_dir)));
    }

    return vec3(0.0);
}

// these functions operate in normal space
#define BRDF(name) vec3 brdf_##name(in vec3 wo, in vec3 wi, in MaterialData material)
#define PDF(name) float pdf_##name(in vec3 wo, in vec3 wi, in MaterialData material)
#define SAMPLE_F(name) vec3 name(in vec3 wo, in MaterialData material, out vec3 wi, out float pdf)

BRDF(diffuse) {
    return material.base_color * INV_PI;
}
PDF(diffuse) {
    return squareToHemisphereCosinePDF(wi);
}
SAMPLE_F(diffuse) {
    vec2 xi = random_pcg3d(uvec3(gl_LaunchIDEXT.xy, payload.iteration + payload.level)).xy;
    wi = squareToHemisphereCosine(xi);
    pdf = pdf_diffuse(wo, wi, material);
    return brdf_diffuse(wo, wi, material);
}

BRDF(specular) {
    return material.base_color / abs(wi.z);
}
PDF(specular) {
    return 1.0;
}
SAMPLE_F(specular) {
    wi = vec3(wo.x, wo.y, -wo.z);
    pdf = pdf_specular(wo, wi, material);

    float absCosTheta = abs(wi.z);
    if (absCosTheta <= Epsilon) {
        // total internal reflection
        return vec3(0.0);
    }
    return brdf_specular(wo, wi, material);
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
        payload.Li = material.emissive_color * material.emission_intensity;
        payload.done = true;
    } else {
        vec3 wo = worldToNormalSpace(nW) * (-gl_WorldRayDirectionEXT);
        float weight_light = 0.0;
        float weight_brdf = 0.0;
        vec3 light_Li = vec3(0.0);
        int light_idx = 0;

        if (num_lights > 0)
        {
            // direct lighting
            vec3 wi;
            vec3 wiW; float pdf;
            vec3 Li = sample_light(posW, nW, wiW, pdf, light_idx);
            wi = worldToNormalSpace(nW) * wiW;
            if (Li != vec3(0.0) && pdf != 0.0f) {
                Li *= brdf_diffuse(wo, wi, material);
                weight_light = power_heuristic(1.0, pdf, 1.0, pdf_diffuse(wo, wi, material));
                light_Li = Li;
            }
        }
        if (payload.direct_lighting_only) {
            payload.Li = light_Li;
            payload.done = true;
        } else {
            vec3 wi; float pdf;
            vec3 Li = diffuse(wo, material, wi, pdf);
            vec3 wiW = normalToWorldSpace(nW) * wi;

            if (pdf == 0.0) {
                payload.invalid = true;
                return;
            }
            
            Li *= abs(dot(wiW, nW)) / pdf;
            weight_brdf = power_heuristic(1.0, pdf, 1.0, pdf_light(posW, nW, wiW, light_idx));
            payload.Li = weight_light * light_Li + weight_brdf * Li;
            payload.pos = posW;
            payload.wi = wiW;
            payload.done = false;
        }

    }
} 