struct Material {
    float3 diffuse_color;
    float metallic;
    float roughness;
    float opacity;
    float2 _pad;
};

struct Uniforms {
    float4x4 mvp;
    float4x4 model;
    float3 sun_dir;
    float time;
    uint material_index;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> u;

[[vk::binding(1, 0)]]
StructuredBuffer<Material> materials;

struct VsIn {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float3 color    : COLOR;
    float2 uv       : TEXCOORD;
};

struct VsOut {
    float4 position     : SV_Position;
    float3 world_normal : NORMAL;
    float3 color        : COLOR;
    float2 uv           : TEXCOORD;
};

[shader("vertex")]
VsOut vs_main(VsIn input) {
    VsOut output;
    output.position = mul(u.mvp, float4(input.position, 1.0));
    output.world_normal = normalize(mul(u.model, float4(input.normal, 0.0)).xyz);
    output.color = input.color;
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fs_main(float3 world_normal : NORMAL, float3 color : COLOR) : SV_Target0 {
    float3 n = normalize(world_normal);
    float ndotl = max(dot(n, u.sun_dir), 0.0);

    float3 base_color;
    if (u.material_index == 0xFFFFFFFF) {
        base_color = color;
    } else {
        base_color = materials[u.material_index].diffuse_color;
    }

    float3 lit = ndotl * base_color + 0.15 * base_color;

    return float4(lit, 1.0);
}
