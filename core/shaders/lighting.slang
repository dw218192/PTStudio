import brdf;
import light;
import ltc;
import sampling;
import shadow_sampling;

// ---------------------------------------------------------------------------
// Lighting evaluation — NEE with stochastic light sampling (path tracer)
// ---------------------------------------------------------------------------

float3 evaluate_lighting_nee<B : IBRDF>(
    B brdf,
    StructuredBuffer<Light> lights, uint light_count,
    float3 P, float3 N, float3 V,
    float3 base_color, float metallic, float roughness,
    inout Rng rng,
    out float3 direct_diffuse, out float3 direct_specular
) {
    direct_diffuse = float3(0.0);
    direct_specular = float3(0.0);

    float3 result = float3(0.0);
    for (uint i = 0; i < light_count; i++) {
        Light light = lights[i];

        if (light.type == LIGHT_DOME) continue;

        LightSampleStochastic ls = sample_direct(light, P, rng);
        if (!ls.valid || ls.pdf <= 0.0) continue;

        float3 wi = normalize(ls.position - P);
        float NdotL = dot(N, wi);
        if (NdotL <= 0.0) continue;

        ShadingGeometry g = ShadingGeometry::from_world(N, wi, V, roughness);
        float3 f_diffuse = brdf.eval_diffuse(g, base_color, metallic);
        float3 f_specular = brdf.eval_specular(g, base_color, metallic);

        float3 contrib = ls.radiance * NdotL / ls.pdf;
        direct_diffuse += f_diffuse * contrib;
        direct_specular += f_specular * contrib;
        result += (f_diffuse + f_specular) * contrib;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Lighting evaluation — LTC for rect/disk area lights (forward renderer)
// ---------------------------------------------------------------------------

float3 evaluate_lighting_ltc<B : IBRDF>(
    B brdf,
    StructuredBuffer<Light> lights, uint light_count,
    Texture2D<float4> ltc_mat, Texture2D<float2> ltc_amp, SamplerState ltc_sampler,
    float3 P, float3 N, float3 V,
    float3 base_color, float metallic, float roughness,
    StructuredBuffer<ShadowInfo> shadow_infos,
    Texture2DArray<float> shadow_map,
    SamplerState shadow_sampler,
    out float3 direct_diffuse, out float3 direct_specular
) {
    direct_diffuse = float3(0.0);
    direct_specular = float3(0.0);

    float3 result = float3(0.0);
    for (uint i = 0; i < light_count; i++) {
        Light light = lights[i];

        if (light.type == LIGHT_DOME) continue;

        float shadow_vis = 1.0;
        ShadowInfo si = shadow_infos[i];
        if (si.has_shadow != 0) {
            float3 offset_pos = P + N * si.normal_bias;
            float4 light_clip = mul(si.light_vp, float4(offset_pos, 1.0));
            float3 ndc = light_clip.xyz / light_clip.w;
            float3 shadow_uv = float3(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5, ndc.z);
            shadow_vis = compute_shadow_pcf(shadow_map, shadow_sampler, shadow_uv, si.layer, si.texel_size);
        }

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
