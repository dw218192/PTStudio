/// Per-light shadow info (must match C++ ShadowInfo, 80 bytes).
/// One entry per light. Lights without shadows have has_shadow == 0.
public struct ShadowInfo {
    public float4x4 light_vp;   // 64 bytes
    public float texel_size;    //  4 bytes
    public float normal_bias;   //  4 bytes
    public uint has_shadow;     //  4 bytes — 0 = no shadow, 1 = shadow active
    public uint layer;          //  4 bytes — index into shadow map texture array
};

// 5x5 PCF with manual depth comparison
public float compute_shadow_pcf(
    Texture2DArray<float> shadow_map,
    SamplerState shadow_sampler,
    float3 shadow_coord,    // xy = UV in [0,1], z = reference depth in [0,1]
    uint layer,
    float texel_size        // 1.0 / shadow_map_resolution
) {
    float shadow = 0.0;
    float ref_depth = shadow_coord.z;
    // 5x5 grid centered at shadow_coord.xy
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            float2 offset = float2(float(x), float(y)) * texel_size;
            float2 uv = shadow_coord.xy + offset;
            float sampled_depth = shadow_map.SampleLevel(
                shadow_sampler,
                float3(uv, float(layer)),
                0
            ).x;
            shadow += (ref_depth <= sampled_depth) ? 1.0 : 0.0;
        }
    }
    return shadow / 25.0;
}
