// Temporal accumulation of PCSS shadow visibility for the first shadow-casting
// light. Reconstructs world position from screen-space depth, runs the
// existing compute_shadow_pcss for that light, then EMA-blends with the
// previous-frame visibility sample. POC for the area_light_pcss image-diff
// case -- only handles a single light and assumes the camera is stationary
// (no motion-vector reprojection).
//
// Output: R16Float visibility in [0, 1] (1 = fully lit, 0 = fully shadowed).

import shadow_sampling;

struct TemporalVisibilityUniforms {
    float4x4 inv_view_proj;     // 0:   64
    float2 viewport_size;       // 64:  8
    float alpha;                // 72:  4 -- weight of curr sample (1.0 = no blend)
    uint shadow_light_index;    // 76:  4 -- index into shadow_infos
    uint frame_index;           // 80:  4 -- monotonic per-frame counter; jitters
                                //          per-pixel PCSS rotation so EMA has a
                                //          stochastic signal to average.
    uint _pad1;                 // 84:  4
    uint _pad2;                 // 88:  4
    uint _pad3;                 // 92:  4 -> total 96
};

// Bind group 0
[[vk::binding(0, 0)]] [NonFiltering] Texture2D<float> depth_tex;
[[vk::binding(1, 0)]] [NonFiltering] SamplerState     depth_sampler;
[[vk::binding(2, 0)]] StructuredBuffer<ShadowInfo>    shadow_infos;
[[vk::binding(3, 0)]] [NonFiltering] Texture2DArray<float> shadow_map;
[[vk::binding(4, 0)]] [NonFiltering] SamplerState     shadow_sampler;
[[vk::binding(5, 0)]] Texture2D<float>                prev_visibility;
[[vk::binding(6, 0)]] SamplerState                    prev_sampler;
[[vk::binding(7, 0)]] ConstantBuffer<TemporalVisibilityUniforms> u;

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

float3 reconstruct_world_pos(float2 uv, float depth) {
    float2 ndc_xy = float2(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);
    float4 clip = float4(ndc_xy, depth, 1.0);
    float4 world = mul(u.inv_view_proj, clip);
    return world.xyz / world.w;
}

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD) : SV_Target0 {
    float depth = depth_tex.SampleLevel(depth_sampler, uv, 0).x;

    // Reconstruct world position and project to shadow_uv for every pixel --
    // including background -- before the background branch: ddx/ddy(shadow_uv)
    // below must evaluate under uniform control flow.
    float3 world_pos = reconstruct_world_pos(uv, depth);

    ShadowInfo si = shadow_infos[u.shadow_light_index];
    float4 light_clip = mul(si.light_vp, float4(world_pos, 1.0));
    float3 ndc = light_clip.xyz / light_clip.w;
    float3 shadow_uv = float3(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5, ndc.z);

    float3 ddx_shadow_uv = ddx(shadow_uv);
    float3 ddy_shadow_uv = ddy(shadow_uv);

    // Background pixels: hold previous visibility as-is so we don't bias the
    // history toward fully-lit on subsequent reads of stale UVs.
    if (depth >= 1.0) {
        return float4(prev_visibility.SampleLevel(prev_sampler, uv, 0).x, 0, 0, 0);
    }

    float curr_vis = 1.0;

    // Mirror lighting.slang's frustum guards: skip projection for points
    // behind the light or outside the [0,1]^3 shadow frustum. ClampToEdge on
    // the shadow sampler turns out-of-frustum samples into false shadows,
    // so we MUST gate before calling compute_shadow_pcss.
    if (si.has_shadow != 0 && light_clip.w > 0.0) {
        if (shadow_uv.x >= 0.0 && shadow_uv.x <= 1.0 &&
            shadow_uv.y >= 0.0 && shadow_uv.y <= 1.0 &&
            shadow_uv.z >= 0.0 && shadow_uv.z <= 1.0) {
            // Receiver-plane depth-bias gradient -- required so Poisson
            // taps that wander to other UVs predict the floor's own
            // depth at the tap's UV instead of misreading it as a
            // blocker via the receiver's center-pixel NDC.
            float2 grad = compute_receiver_plane_grad(ddx_shadow_uv, ddy_shadow_uv);
            curr_vis = compute_shadow_pcss(
                shadow_map, shadow_sampler, shadow_uv, si, uv, u.frame_index, grad);
        }
    }

    float prev_vis = prev_visibility.SampleLevel(prev_sampler, uv, 0).x;
    float blended = u.alpha * curr_vis + (1.0 - u.alpha) * prev_vis;
    return float4(blended, 0, 0, 0);
}
