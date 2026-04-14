struct Uniforms {
    float4x4 mvp;
    float4 color;
};

[[vk::binding(0, 0)]]
[DynamicBuffer] ConstantBuffer<Uniforms> u;

struct VsIn {
    float3 position : POSITION;
};

[shader("vertex")]
float4 vs_main(VsIn input) : SV_Position {
    return mul(u.mvp, float4(input.position, 1.0));
}

[shader("fragment")]
float4 fs_main() : SV_Target0 {
    return u.color;
}
