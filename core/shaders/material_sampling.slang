import light;

public static const uint k_invalid_tex = 0xFFFFFFFF;

public bool has_texture(uint tex_id) {
    return tex_id != k_invalid_tex;
}

public struct MaterialSample {
    public float3 base_color;
    public float metallic;
    public float roughness;
};

/// Read material properties and sample textures (LOD-explicit version for compute shaders).
public MaterialSample sample_material_lod(
    Material mat, float2 uv,
    Texture2DArray<float4> textures, SamplerState samp, float lod)
{
    MaterialSample s;
    s.base_color = mat.diffuse_color;
    s.metallic = mat.metallic;
    s.roughness = mat.roughness;

    if (has_texture(mat.diffuse_tex)) {
        float4 tc = textures.SampleLevel(samp, float3(uv, float(mat.diffuse_tex)), lod);
        s.base_color = pow(tc.rgb, float3(2.2));
    }

    if (has_texture(mat.metallic_tex)) {
        uint ch = mat.tex_channels & 0x3;
        s.metallic = textures.SampleLevel(samp, float3(uv, float(mat.metallic_tex)), lod)[ch];
    }

    if (has_texture(mat.roughness_tex)) {
        uint ch = (mat.tex_channels >> 2) & 0x3;
        s.roughness = textures.SampleLevel(samp, float3(uv, float(mat.roughness_tex)), lod)[ch];
    }

    return s;
}

/// Read material properties and sample textures (implicit-LOD version for fragment shaders).
public MaterialSample sample_material(
    Material mat, float2 uv,
    Texture2DArray<float4> textures, SamplerState samp)
{
    MaterialSample s;
    s.base_color = mat.diffuse_color;
    s.metallic = mat.metallic;
    s.roughness = mat.roughness;

    if (has_texture(mat.diffuse_tex)) {
        float4 tc = textures.Sample(samp, float3(uv, float(mat.diffuse_tex)));
        s.base_color = pow(tc.rgb, float3(2.2));
    }

    if (has_texture(mat.metallic_tex)) {
        uint ch = mat.tex_channels & 0x3;
        s.metallic = textures.Sample(samp, float3(uv, float(mat.metallic_tex)))[ch];
    }

    if (has_texture(mat.roughness_tex)) {
        uint ch = (mat.tex_channels >> 2) & 0x3;
        s.roughness = textures.Sample(samp, float3(uv, float(mat.roughness_tex)))[ch];
    }

    return s;
}

/// Apply a normal map given a pre-computed tangent frame.
public float3 apply_normal_map(
    float3 N, float3 T, float3 B,
    Material mat, float2 uv,
    Texture2DArray<float4> textures, SamplerState samp, float lod)
{
    if (!has_texture(mat.normal_tex)) return N;
    float3 ns = textures.SampleLevel(samp, float3(uv, float(mat.normal_tex)), lod).rgb;
    ns = ns * 2.0 - 1.0;
    return normalize(T * ns.x + B * ns.y + N * ns.z);
}

/// Apply a normal map using screen-space derivatives for the tangent frame (fragment shaders only).
public float3 apply_normal_map_screen(
    float3 N, float3 world_pos, float2 uv,
    Material mat,
    Texture2DArray<float4> textures, SamplerState samp)
{
    if (!has_texture(mat.normal_tex)) return N;
    float3 ns = textures.Sample(samp, float3(uv, float(mat.normal_tex))).rgb;
    ns = ns * 2.0 - 1.0;
    float3 dp1 = ddx(world_pos);
    float3 dp2 = ddy(world_pos);
    float2 duv1 = ddx(uv);
    float2 duv2 = ddy(uv);
    float3 T = normalize(dp1 * duv2.y - dp2 * duv1.y);
    float3 B = normalize(dp2 * duv1.x - dp1 * duv2.x);
    return normalize(T * ns.x + B * ns.y + N * ns.z);
}
