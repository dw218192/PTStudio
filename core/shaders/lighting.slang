import brdf;
import light;
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
