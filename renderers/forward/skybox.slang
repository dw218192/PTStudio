struct SkyboxUniforms {
    float4x4 inv_vp;
    float3 camera_pos;
    float _pad0;
    float3 dome_modulation;
    float _pad1;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SkyboxUniforms> u;

[[vk::binding(1, 0)]]
TextureCube<float4> env_cubemap;

[[vk::binding(2, 0)]]
SamplerState env_sampler;

struct VsOut {
    float4 position : SV_Position;
    float2 ndc : TEXCOORD;
};

// Fullscreen triangle from vertex_id (no vertex buffer needed)
[shader("vertex")]
VsOut vs_main(uint vertex_id : SV_VertexID) {
    VsOut output;
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(uv * 2.0 - 1.0, 1.0, 1.0);  // Z=1.0 at far plane
    output.ndc = uv * 2.0 - 1.0;
    return output;
}

[shader("fragment")]
float4 fs_main(float2 ndc : TEXCOORD) : SV_Target0 {
    float4 near_pt = mul(u.inv_vp, float4(ndc, 0.0, 1.0));
    float4 far_pt  = mul(u.inv_vp, float4(ndc, 1.0, 1.0));
    float3 dir = normalize(far_pt.xyz / far_pt.w - near_pt.xyz / near_pt.w);
    float3 color = env_cubemap.SampleLevel(env_sampler, dir, 0.0).rgb;
    color *= u.dome_modulation;
    return float4(color, 1.0);
}
