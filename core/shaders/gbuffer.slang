struct GBufferUniforms {
    float4x4 mvp;
    float4x4 model_view;
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
};

[shader("vertex")]
VsOut vs_main(VsIn input) {
    VsOut output;
    output.position = mul(u.mvp, float4(input.position, 1.0));
    output.view_normal = normalize(mul(u.model_view, float4(input.normal, 0.0)).xyz);
    return output;
}

struct FsOut {
    float2 normal : SV_Target0;
};

[shader("fragment")]
FsOut fs_main(float3 view_normal : VIEW_NORMAL) {
    FsOut output;
    float3 n = normalize(view_normal);
    output.normal = float2(n.x, n.y);
    return output;
}
