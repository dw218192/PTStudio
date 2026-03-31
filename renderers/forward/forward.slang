import lighting;
import brdf;
import ibl;
import light;
import material_sampling;
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
    float3 ibl_dome_modulation;
    uint ibl_mip_count;
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

[[vk::binding(6, 0)]]
Texture2DArray<float4> scene_textures;

[[vk::binding(7, 0)]]
SamplerState scene_sampler;

// Bind group 1: shadow data
[[vk::binding(0, 1)]] StructuredBuffer<ShadowInfo> shadow_infos;
[[vk::binding(1, 1)]] Texture2DArray<float> shadow_map;
[[vk::binding(2, 1)]] SamplerState shadow_sampler;

// Bind group 2: IBL
[[vk::binding(0, 2)]] TextureCube<float4> ibl_prefiltered;
[[vk::binding(1, 2)]] TextureCube<float4> ibl_irradiance;
[[vk::binding(2, 2)]] Texture2D<float4> ibl_brdf_lut;
[[vk::binding(3, 2)]] SamplerState ibl_sampler;

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
    float4 color           : SV_Target0;
    float4 direct_diffuse  : SV_Target1;
    float4 direct_specular : SV_Target2;
    float4 ibl_diffuse     : SV_Target3;
    float4 ibl_specular    : SV_Target4;
    float4 prefiltered_env : SV_Target5;
    float4 brdf_lut        : SV_Target6;
};
#else
struct FsOut {
    float4 color : SV_Target0;
};
#endif

[shader("fragment")]
FsOut fs_main(float3 world_normal : NORMAL, float3 world_pos : WORLD_POS,
              float3 color : COLOR, float2 uv : TEXCOORD) {
    float3 N = normalize(world_normal);
    float3 V = normalize(u.camera_pos - world_pos);

    Material mat = materials[u.material_index];
    MaterialSample sm = sample_material(mat, uv, scene_textures, scene_sampler);
    float3 albedo = sm.base_color;
    float mat_metallic = sm.metallic;
    float mat_roughness = sm.roughness;
    N = apply_normal_map_screen(N, world_pos, uv, mat, scene_textures, scene_sampler);

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

    float3 ibl_diff, ibl_spec, ibl_prefilt, ibl_brdf_scale;
    float3 ibl = evaluate_ibl(ibl_prefiltered, ibl_irradiance, ibl_brdf_lut, ibl_sampler,
                               N, V, albedo, mat_metallic, mat_roughness, u.ibl_mip_count,
                               ibl_diff, ibl_spec, ibl_prefilt, ibl_brdf_scale);
    lit += ibl * u.ibl_dome_modulation;

    lit += mat.emissive_color;

    FsOut o;
    o.color = float4(lit, 1.0);
#ifndef NO_DEBUG_TARGETS
    o.direct_diffuse  = float4(direct_diffuse, 1.0);
    o.direct_specular = float4(direct_specular, 1.0);
    o.ibl_diffuse     = float4(ibl_diff * u.ibl_dome_modulation, 1.0);
    o.ibl_specular    = float4(ibl_spec * u.ibl_dome_modulation, 1.0);
    o.prefiltered_env = float4(ibl_prefilt * u.ibl_dome_modulation, 1.0);
    o.brdf_lut        = float4(ibl_brdf_scale, 1.0);
#endif
    return o;
}
