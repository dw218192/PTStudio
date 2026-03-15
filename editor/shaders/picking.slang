struct Uniforms {
    float4x4 mvp;
    uint object_id;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> u;

struct VsIn {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float3 color    : COLOR;
    float2 uv       : TEXCOORD;
};

[shader("vertex")]
float4 vs_main(VsIn input) : SV_Position {
    return mul(u.mvp, float4(input.position, 1.0));
}

[shader("fragment")]
uint fs_main() : SV_Target0 {
    return u.object_id;
}
