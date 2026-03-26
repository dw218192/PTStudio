import lighting;
import brdf;
import light;
import shadow_sampling;

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

[[vk::binding(3, 0)]]
Texture2D<float4> ltc_mat;

[[vk::binding(4, 0)]]
Texture2D<float2> ltc_amp;

[[vk::binding(5, 0)]]
SamplerState ltc_sampler;

// Bind group 1: shadow data
[[vk::binding(0, 1)]] StructuredBuffer<ShadowInfo> shadow_infos;
[[vk::binding(1, 1)]] Texture2DArray<float> shadow_map;
[[vk::binding(2, 1)]] SamplerState shadow_sampler;

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

#ifndef NO_DEBUG_TARGETS
struct FsOut {
    float4 color      : SV_Target0;
    float4 normals    : SV_Target1;
    float4 base_color : SV_Target2;
    float4 direct_diffuse  : SV_Target3;
    float4 direct_specular   : SV_Target4;
};
#else
struct FsOut {
    float4 color : SV_Target0;
};
#endif

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

    float3 direct_diffuse, direct_specular;
    float3 lit = evaluate_lighting_ltc(
        brdf,
        lights,
        u.light_count,
        ltc_mat, ltc_amp, ltc_sampler,
        world_pos, N, V, albedo, mat_metallic, mat_roughness,
        shadow_infos, shadow_map, shadow_sampler,
        direct_diffuse, direct_specular
    );

    FsOut o;
    o.color = float4(lit, 1.0);
#ifndef NO_DEBUG_TARGETS
    o.normals    = float4(N * 0.5 + 0.5, 1.0);
    o.base_color = float4(albedo, 1.0);
    o.direct_diffuse  = float4(direct_diffuse, 1.0);
    o.direct_specular = float4(direct_specular, 1.0);
#endif
    return o;
}
