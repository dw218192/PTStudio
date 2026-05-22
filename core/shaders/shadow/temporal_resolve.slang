// Temporal resolve pass. Reads the raw shadow-visibility texture produced by
// the shadow_visibility (gen) pass plus the previous frame's resolved history,
// applies a 3x3 neighborhood variance clamp to the history sample, then
// EMA-blends it with the raw sample.
//
// History is sampled at the same screen UV -- motion-vector reprojection is a
// separate ticket (motion-vector-reprojection).
//
// Output: R16Float resolved visibility in [0, 1].

import shadow_sampling;

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

    float history = history_visibility.SampleLevel(history_sampler, uv, 0).x;
    float clamped_history = clamp_history_to_neighborhood(history, mean, stddev, u.gamma);

    float resolved = temporal_ema_blend(curr, clamped_history, u.alpha);
    return float4(resolved, 0, 0, 0);
}
