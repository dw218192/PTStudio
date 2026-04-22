import brdf;
import light;
import light_iteration;
import ltc;
import shadow_sampling;

// ---------------------------------------------------------------------------
// Lighting evaluation -- LTC for rect/disk area lights (forward renderer)
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
    float2 screen_uv,
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

        float shadow_vis = 1.0;
        ShadowInfo si = shadow_infos[i];
        if (si.has_shadow != 0) {
            float3 offset_pos = P + N * si.normal_bias;
            float4 light_clip = mul(si.light_vp, float4(offset_pos, 1.0));
            // Skip shadow for points behind the light (w <= 0 flips the
            // perspective divide) or outside the shadow frustum UV range.
            if (light_clip.w > 0.0) {
                float3 ndc = light_clip.xyz / light_clip.w;
                float3 shadow_uv = float3(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5, ndc.z);
                if (shadow_uv.x >= 0.0 && shadow_uv.x <= 1.0 &&
                    shadow_uv.y >= 0.0 && shadow_uv.y <= 1.0 &&
                    shadow_uv.z >= 0.0 && shadow_uv.z <= 1.0) {
                    shadow_vis = compute_shadow_pcss(
                        shadow_map, shadow_sampler, shadow_uv, si, screen_uv);
                }
            }
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
