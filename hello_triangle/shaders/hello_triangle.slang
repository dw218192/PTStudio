struct Uniforms {
    float4x4 mvp;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> u;

struct VsIn {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float3 color    : COLOR;
};

struct VsOut {
    float4 position : SV_Position;
    float3 color    : COLOR;
};

[shader("vertex")]
VsOut vs_main(VsIn input) {
    VsOut output;
    output.position = mul(u.mvp, float4(input.position, 1.0));
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(float3 color : COLOR) : SV_Target {
    return float4(color, 1.0);
}
