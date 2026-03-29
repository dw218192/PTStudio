import brdf;

public float3 evaluate_ibl(
    TextureCube<float4> prefiltered_env,
    TextureCube<float4> irradiance_map,
    Texture2D<float4> brdf_lut,
    SamplerState ibl_sampler,
    float3 N, float3 V,
    float3 base_color, float metallic, float roughness,
    uint num_prefilter_mips,
    out float3 out_diffuse, out float3 out_specular
) {
    float3 F0 = lerp(float3(0.04), base_color, metallic);
    float NdotV = max(dot(N, V), 0.001);

    // Diffuse: irradiance * kD * albedo
    float3 F_nv = F_Schlick(NdotV, F0);
    float3 kD = (1.0 - F_nv) * (1.0 - metallic);
    float3 irradiance = irradiance_map.Sample(ibl_sampler, N).rgb;
    out_diffuse = kD * (base_color / PI) * irradiance;

    // Specular: prefiltered env * (F0 * lut.r + lut.g)
    float3 R = reflect(-V, N);
    float lod = roughness * float(num_prefilter_mips - 1);
    float3 prefiltered = prefiltered_env.SampleLevel(ibl_sampler, R, lod).rgb;
    float2 env_brdf = brdf_lut.Sample(ibl_sampler, float2(NdotV, roughness)).rg;
    out_specular = prefiltered * (F0 * env_brdf.x + env_brdf.y);

    return out_diffuse + out_specular;
}
