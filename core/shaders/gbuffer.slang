struct GBufferUniforms {
    float4x4 mvp;
    float4x4 model_view;
    float4x4 prev_mvp;
};

[[vk::binding(0, 0)]]
[DynamicBuffer] ConstantBuffer<GBufferUniforms> u;

struct VsIn {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float3 color    : COLOR;
    float2 uv       : TEXCOORD;
};

struct VsOut {
    float4 position    : SV_Position;
    float3 view_normal : VIEW_NORMAL;
    float4 curr_clip   : CURR_CLIP;
    float4 prev_clip   : PREV_CLIP;
};

[shader("vertex")]
VsOut vs_main(VsIn input) {
    VsOut output;
    float4 local_pos = float4(input.position, 1.0);
    float4 curr_clip = mul(u.mvp, local_pos);
    output.position = curr_clip;
    output.view_normal = normalize(mul(u.model_view, float4(input.normal, 0.0)).xyz);
    output.curr_clip = curr_clip;
    output.prev_clip = mul(u.prev_mvp, local_pos);
    return output;
}

struct FsOut {
    float2 normal : SV_Target0;
    // Motion vector in UV space: prev_uv - curr_uv. Consumers add this to
    // their current-frame UV to sample previous-frame data (history
    // reprojection). Static camera + static geometry yields ~zero motion.
    float2 motion : SV_Target1;
};

[shader("fragment")]
FsOut fs_main(VsOut input) {
    FsOut output;
    float3 n = normalize(input.view_normal);
    output.normal = float2(n.x, n.y);

    // WebGPU clip-to-UV: uv.x = ndc.x*0.5 + 0.5, uv.y = 0.5 - ndc.y*0.5
    // (the rasterizer y-flip the temporal_resolve VS also uses).
    float curr_w = max(input.curr_clip.w, 1e-6);
    float2 curr_ndc = input.curr_clip.xy / curr_w;
    float2 curr_uv = float2(curr_ndc.x * 0.5 + 0.5, 0.5 - curr_ndc.y * 0.5);

    // prev_clip.w can go non-positive when the surface was behind the camera
    // last frame; the consumer's bounds check on (uv + motion) rejects those.
    float prev_w = max(input.prev_clip.w, 1e-6);
    float2 prev_ndc = input.prev_clip.xy / prev_w;
    float2 prev_uv = float2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);

    output.motion = prev_uv - curr_uv;
    return output;
}
