struct ShadowModel {
    float4x4 model;
};

struct ShadowLightVP {
    float4x4 light_vp;
};

[[vk::binding(0, 0)]]
[DynamicBuffer] ConstantBuffer<ShadowModel> obj;

[[vk::binding(1, 0)]]
[DynamicBuffer] ConstantBuffer<ShadowLightVP> light;

struct VsIn {
    float3 position : POSITION;
};

struct VsOut {
    float4 position : SV_Position;
};

[shader("vertex")]
VsOut vs_main(VsIn input) {
    VsOut output;
    output.position = mul(light.light_vp, mul(obj.model, float4(input.position, 1.0)));
    return output;
}
