import brdf_lib;
import light_lib;
import light_iteration_lib;
import ltc_lib;

// ---------------------------------------------------------------------------
// Lighting evaluation -- LTC for rect/disk area lights (forward renderer)
// ---------------------------------------------------------------------------

float3 evaluate_lighting_ltc<B : IBRDF>(
    B brdf,
    StructuredBuffer<Light> lights, uint light_count,
    Texture2D<float4> ltc_mat, Texture2D<float2> ltc_amp, SamplerState ltc_sampler,
    float3 P, float3 N, float3 V,
    float3 base_color, float metallic, float roughness,
    float2 screen_uv,
    uint shadow_light_index,
    float resolved_shadow_vis,
    out float3 direct_diffuse, out float3 direct_specular
) {
    direct_diffuse = float3(0.0);
    direct_specular = float3(0.0);

    float3 result = float3(0.0);
    uint visible_count = get_light_count(light_count, screen_uv);
    for (uint iter = 0; iter < visible_count; iter++) {
        uint i = get_light_index(iter, screen_uv);
        Light light = lights[i];

        if (light.type == LIGHT_DOME) continue;

        // Single-light shadow scope: the one shadow-casting light's visibility
        // is supplied by the resolved shadow-visibility texture (gen + resolve
        // pipeline; frustum guards live in the gen pass). Every other light --
        // including secondary shadow casters -- renders unshadowed; multi-light
        // shadows are deferred to clustered lighting.
        float shadow_vis = (i == shadow_light_index) ? resolved_shadow_vis : 1.0;

        if (light.type == LIGHT_RECT || light.type == LIGHT_DISK) {
            float3 diff, spec;
            if (light.type == LIGHT_RECT) {
                ltc_evaluate_rect(light, P, N, V, roughness, base_color, metallic,
                                  ltc_mat, ltc_amp, ltc_sampler, diff, spec);
            } else {
                ltc_evaluate_disk(light, P, N, V, roughness, base_color, metallic,
                                  ltc_mat, ltc_amp, ltc_sampler, diff, spec);
            }
            diff *= shadow_vis;
            spec *= shadow_vis;
            result += diff + spec;
            direct_diffuse += diff;
            direct_specular += spec;
            continue;
        }

        LightSample ls = sample_representative(light, P, N, V, roughness);
        if (!ls.valid) continue;

        ShadingGeometry gd = ShadingGeometry::from_world(N, ls.diffuse_L, V, roughness);
        ShadingGeometry gs = ShadingGeometry::from_world(N, ls.spec_L, V, ls.spec_roughness);
        float3 brdf_diffuse = brdf.eval_diffuse(gd, base_color, metallic);
        float3 brdf_specular = brdf.eval_specular(gs, base_color, metallic);

        float3 radiance_diffuse = light.color * light.intensity * ls.diffuse_attenuation;
        float3 radiance_specular = light.color * light.intensity * ls.spec_attenuation;

        float3 diffuse = brdf_diffuse * max(dot(N, ls.diffuse_L), 0.0) * radiance_diffuse;
        float3 specular = brdf_specular * max(dot(N, ls.spec_L), 0.0) * radiance_specular;

        diffuse *= shadow_vis;
        specular *= shadow_vis;

        result += diffuse + specular;
        direct_diffuse += diffuse;
        direct_specular += specular;
    }
    return result;
}
