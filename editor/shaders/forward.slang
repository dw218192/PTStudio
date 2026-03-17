import lighting;

struct Uniforms {
    float4x4 mvp;
    float4x4 model;
    float3 camera_pos;
    float time;
    uint material_index;
    uint light_count;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> u;

[[vk::binding(1, 0)]]
StructuredBuffer<Material> materials;

[[vk::binding(2, 0)]]
StructuredBuffer<GpuLight> lights;

struct VsIn {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float3 color    : COLOR;
    float2 uv       : TEXCOORD;
};

struct VsOut {
    float4 position     : SV_Position;
    float3 world_normal : NORMAL;
    float3 world_pos    : WORLD_POS;
    float3 color        : COLOR;
    float2 uv           : TEXCOORD;
};

[shader("vertex")]
VsOut vs_main(VsIn input) {
    VsOut output;
    output.position = mul(u.mvp, float4(input.position, 1.0));
    output.world_normal = normalize(mul(u.model, float4(input.normal, 0.0)).xyz);
    output.world_pos = mul(u.model, float4(input.position, 1.0)).xyz;
    output.color = input.color;
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fs_main(float3 world_normal : NORMAL, float3 world_pos : WORLD_POS,
               float3 color : COLOR) : SV_Target0 {
    float3 N = normalize(world_normal);
    float3 V = normalize(u.camera_pos - world_pos);

    float3 base_color;
    float roughness = 0.5;
    if (u.material_index == 0xFFFFFFFF) {
        base_color = color;
    } else {
        base_color = materials[u.material_index].diffuse_color;
        roughness = materials[u.material_index].roughness;
    }

    BlinnPhong brdf;
    float3 result = evaluate_lighting(brdf, lights, u.light_count, world_pos, N, V, base_color, roughness);
    result += 0.03 * base_color;
    return float4(result, 1.0);
}
