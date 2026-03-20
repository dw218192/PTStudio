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
StructuredBuffer<Light> lights;

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

struct FsOut {
    float4 color      : SV_Target0;
    float4 normals    : SV_Target1;
    float4 base_color : SV_Target2;
    float4 roughness  : SV_Target3;
    float4 metallic   : SV_Target4;
};

[shader("fragment")]
FsOut fs_main(float3 world_normal : NORMAL, float3 world_pos : WORLD_POS,
              float3 color : COLOR) {
    float3 N = normalize(world_normal);
    float3 V = normalize(u.camera_pos - world_pos);

    float3 albedo;
    float mat_metallic = 0.0;
    float mat_roughness = 0.5;
    if (u.material_index == 0xFFFFFFFF) {
        albedo = color;
    } else {
        albedo = materials[u.material_index].diffuse_color;
        mat_metallic = materials[u.material_index].metallic;
        mat_roughness = materials[u.material_index].roughness;
    }

    CookTorranceGGX brdf;
    float3 lit = evaluate_lighting(brdf, lights, u.light_count, world_pos, N, V, albedo, mat_metallic, mat_roughness);

    FsOut o;
    o.color      = float4(lit, 1.0);
    o.normals    = float4(N * 0.5 + 0.5, 1.0);
    o.base_color = float4(albedo, 1.0);
    o.roughness  = float4(mat_roughness, mat_roughness, mat_roughness, 1.0);
    o.metallic   = float4(mat_metallic, mat_metallic, mat_metallic, 1.0);
    return o;
}
