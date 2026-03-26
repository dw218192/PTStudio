struct ShadowPerObject {
    float4x4 light_mvp;
};

[[vk::binding(0, 0)]]
ConstantBuffer<ShadowPerObject> u;

struct VsIn {
    float3 position : POSITION;
};

struct VsOut {
    float4 position : SV_Position;
};

[shader("vertex")]
VsOut vs_main(VsIn input) {
    VsOut output;
    output.position = mul(u.light_mvp, float4(input.position, 1.0));
    return output;
}
