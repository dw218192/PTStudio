import brdf;
import light;
import ltc;
import sampling;

// ---------------------------------------------------------------------------
// Top-level lighting evaluation — generic over BRDF model
// ---------------------------------------------------------------------------

float3 evaluate_lighting<B : IBRDF>(
    B brdf,
    StructuredBuffer<Light> lights, uint light_count,
    float3 P, float3 N, float3 V,
    float3 base_color, float metallic, float roughness,
    out float3 direct_diffuse, out float3 direct_specular
) {
    direct_diffuse = float3(0.0);
    direct_specular = float3(0.0);

    float3 result = float3(0.0);
    for (uint i = 0; i < light_count; i++) {
        Light light = lights[i];

        if (light.type == LIGHT_DOME) {
            result += light.color * light.intensity * base_color * 0.2;
            continue;
        }

        LightSample ls = light.sample(P, N, V, roughness);
        if (!ls.valid) continue;

        ShadingGeometry gd = ShadingGeometry::from_world(N, ls.diffuse_L, V, roughness);
        ShadingGeometry gs = ShadingGeometry::from_world(N, ls.spec_L, V, ls.spec_roughness);
        float3 brdf_diffuse = brdf.eval_diffuse(gd, base_color, metallic);
        float3 brdf_specular = brdf.eval_specular(gs, base_color, metallic);

        float3 radiance_diffuse = light.color * light.intensity * ls.diffuse_attenuation;
        float3 radiance_specular = light.color * light.intensity * ls.spec_attenuation;

        float3 diffuse = brdf_diffuse * max(dot(N, ls.diffuse_L), 0.0) * radiance_diffuse;
        float3 specular = brdf_specular * max(dot(N, ls.spec_L), 0.0) * radiance_specular;

        result += diffuse + specular;
        direct_diffuse += diffuse;
        direct_specular += specular;
    }
    return result;
}

// ---------------------------------------------------------------------------
// LTC-enhanced lighting evaluation — uses LTC for rect/disk area lights
// ---------------------------------------------------------------------------

float3 evaluate_lighting_ltc<B : IBRDF>(
    B brdf,
    StructuredBuffer<Light> lights, uint light_count,
    Texture2D<float4> ltc_mat, Texture2D<float2> ltc_amp, SamplerState ltc_sampler,
    float3 P, float3 N, float3 V,
    float3 base_color, float metallic, float roughness,
    out float3 direct_diffuse, out float3 direct_specular
) {
    direct_diffuse = float3(0.0);
    direct_specular = float3(0.0);

    float3 result = float3(0.0);
    for (uint i = 0; i < light_count; i++) {
        Light light = lights[i];

        if (light.type == LIGHT_DOME) {
            result += light.color * light.intensity * base_color * 0.2;
            continue;
        }

        // Use LTC for rect and disk area lights
        if (light.type == LIGHT_RECT || light.type == LIGHT_DISK) {
            float3 diff, spec;
            if (light.type == LIGHT_RECT) {
                ltc_evaluate_rect(light, P, N, V, roughness, base_color, metallic,
                                  ltc_mat, ltc_amp, ltc_sampler, diff, spec);
            } else {
                ltc_evaluate_disk(light, P, N, V, roughness, base_color, metallic,
                                  ltc_mat, ltc_amp, ltc_sampler, diff, spec);
            }
            result += diff + spec;
            direct_diffuse += diff;
            direct_specular += spec;
            continue;
        }

        // Distant + sphere: use representative-point as before
        LightSample ls = light.sample(P, N, V, roughness);
        if (!ls.valid) continue;

        ShadingGeometry gd = ShadingGeometry::from_world(N, ls.diffuse_L, V, roughness);
        ShadingGeometry gs = ShadingGeometry::from_world(N, ls.spec_L, V, ls.spec_roughness);
        float3 brdf_diffuse = brdf.eval_diffuse(gd, base_color, metallic);
        float3 brdf_specular = brdf.eval_specular(gs, base_color, metallic);

        float3 radiance_diffuse = light.color * light.intensity * ls.diffuse_attenuation;
        float3 radiance_specular = light.color * light.intensity * ls.spec_attenuation;

        float3 diffuse = brdf_diffuse * max(dot(N, ls.diffuse_L), 0.0) * radiance_diffuse;
        float3 specular = brdf_specular * max(dot(N, ls.spec_L), 0.0) * radiance_specular;

        result += diffuse + specular;
        direct_diffuse += diffuse;
        direct_specular += specular;
    }
    return result;
}
