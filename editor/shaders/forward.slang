struct Uniforms {
    float4x4 mvp;
    float4x4 model;
    float3 sun_dir;
    float time;
    uint object_id;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> u;

struct VsIn {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float3 color    : COLOR;
};

struct VsOut {
    float4 position   : SV_Position;
    float3 world_normal : NORMAL;
    float3 color      : COLOR;
};

[shader("vertex")]
VsOut vs_main(VsIn input) {
    VsOut output;
    output.position = mul(u.mvp, float4(input.position, 1.0));
    output.world_normal = normalize(mul(u.model, float4(input.normal, 0.0)).xyz);
    output.color = input.color;
    return output;
}

struct FsOut {
    float4 color    : SV_Target0;
    uint   objectId : SV_Target1;
};

[shader("fragment")]
FsOut fs_main(float3 world_normal : NORMAL, float3 color : COLOR) {
    float3 n = normalize(world_normal);
    float ndotl = max(dot(n, u.sun_dir), 0.0);
    float3 lit = ndotl * color + 0.15 * color;

    FsOut output;
    output.color = float4(lit, 1.0);
    output.objectId = u.object_id;
    return output;
}
