// Temporal resolve pass. Reads the raw shadow-visibility texture produced by
// the shadow_visibility (gen) pass plus the previous frame's resolved history,
// reprojects the history sample using the G-buffer motion vector, applies a
// 3x3 neighborhood variance clamp, then EMA-blends with the raw sample.
//
// Motion vector convention: motion = prev_uv - curr_uv. The reprojected
// history UV is `uv + motion`. Pixels whose reprojected UV falls outside
// [0,1] (disocclusion / off-screen reprojection) bypass the history blend
// entirely (alpha = 1) -- there's no valid history to integrate, so we take
// the raw current sample and rebuild from there.
//
// Output: R16Float resolved visibility in [0, 1].

import shadow_sampling_lib;

struct TemporalResolveUniforms {
    float2 viewport_size;  // 0:  8
    float alpha;           // 8:  4 -- EMA weight of the raw sample (1 = no blend)
    float gamma;           // 12: 4 -- variance-clamp width, in stddevs
};                         // total: 16

// Bind group 0
[[vk::binding(0, 0)]] [NonFiltering] Texture2D<float> raw_visibility;
[[vk::binding(1, 0)]] [NonFiltering] SamplerState     raw_sampler;
[[vk::binding(2, 0)]] Texture2D<float>                history_visibility;
[[vk::binding(3, 0)]] SamplerState                    history_sampler;
[[vk::binding(4, 0)]] ConstantBuffer<TemporalResolveUniforms> u;
[[vk::binding(5, 0)]] [NonFiltering] Texture2D<float2> motion_vector;
[[vk::binding(6, 0)]] [NonFiltering] SamplerState      motion_sampler;

struct VsOut {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD;
};

[shader("vertex")]
VsOut vs_main(uint vertex_id : SV_VertexID) {
    VsOut output;
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = float2(uv.x, 1.0 - uv.y);
    return output;
}

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD) : SV_Target0 {
    float curr = raw_visibility.SampleLevel(raw_sampler, uv, 0).x;

    // Neighborhood variance clamp: reject stale history that diverges from the
    // current frame's local visibility distribution before the EMA blend.
    float2 texel = 1.0 / u.viewport_size;
    float mean;
    float stddev;
    neighborhood_visibility_stats(raw_visibility, raw_sampler, uv, texel, mean, stddev);

    // Motion-vector reprojection: history was rasterized at last frame's
    // screen positions. Step from `uv` to where this surface was last frame.
    float2 motion = motion_vector.SampleLevel(motion_sampler, uv, 0).xy;
    float2 history_uv = uv + motion;

    // Off-screen reprojection -> no valid history. Fall through to alpha=1
    // (raw sample only) instead of clamping at the edge, which would smear
    // the texture border into disoccluded pixels.
    bool valid = all(history_uv >= float2(0.0, 0.0)) && all(history_uv <= float2(1.0, 1.0));
    if (!valid) {
        return float4(curr, 0, 0, 0);
    }

    float history = history_visibility.SampleLevel(history_sampler, history_uv, 0).x;
    float clamped_history = clamp_history_to_neighborhood(history, mean, stddev, u.gamma);

    float resolved = temporal_ema_blend(curr, clamped_history, u.alpha);
    return float4(resolved, 0, 0, 0);
}
