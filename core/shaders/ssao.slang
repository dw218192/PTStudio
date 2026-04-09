struct SSAOUniforms {
    float4x4 projection;
    float4x4 inv_projection;
    float2 viewport_size;
    float radius;
    float bias;
    float intensity;
    int sample_count;
    uint _pad0;
    uint _pad1;
};

// GBuffer consumer slots (0-3)
[[vk::binding(0, 0)]] Texture2D<float>  depth_tex;
[[vk::binding(1, 0)]] SamplerState      depth_sampler;
[[vk::binding(2, 0)]] Texture2D<float4> normals_tex;
[[vk::binding(3, 0)]] SamplerState      linear_sampler;
// SSAO-specific (4-7)
[[vk::binding(4, 0)]] ConstantBuffer<SSAOUniforms> u;
[[vk::binding(5, 0)]] Texture2D<float4> noise_tex;
[[vk::binding(6, 0)]] SamplerState      noise_sampler;
[[vk::binding(7, 0)]] StructuredBuffer<float4> kernel;

struct VsOut {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD;
};

[shader("vertex")]
VsOut vs_main(uint vertex_id : SV_VertexID) {
    VsOut output;
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = float2(uv.x, 1.0 - uv.y);
    return output;
}

// Reconstruct view-space position from UV and depth.
float3 reconstruct_view_pos(float2 uv, float depth) {
    // UV → NDC
    float2 ndc_xy = float2(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);
    float4 clip = float4(ndc_xy, depth, 1.0);
    float4 view = mul(u.inv_projection, clip);
    return view.xyz / view.w;
}

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD) : SV_Target0 {
    // 1. Sample depth and reconstruct view-space position
    float depth = depth_tex.SampleLevel(depth_sampler, uv, 0).x;
    if (depth >= 1.0) {
        return float4(1.0, 0.0, 0.0, 0.0);
    }
    float3 frag_pos = reconstruct_view_pos(uv, depth);

    // 2. Reconstruct view-space normal from RG16Float (xy stored, z derived)
    float2 n_xy = normals_tex.SampleLevel(linear_sampler, uv, 0).xy;
    float n_z = sqrt(max(0.0, 1.0 - n_xy.x * n_xy.x - n_xy.y * n_xy.y));
    float3 normal = normalize(float3(n_xy, n_z));

    // 3. Sample tiled noise for random kernel rotation
    float2 noise_uv = uv * u.viewport_size / 4.0;
    float3 random_vec = noise_tex.SampleLevel(noise_sampler, noise_uv, 0).xyz * 2.0 - 1.0;

    // 4. Build TBN via Gram-Schmidt
    float3 tangent   = normalize(random_vec - normal * dot(random_vec, normal));
    float3 bitangent = cross(normal, tangent);
    float3x3 TBN     = float3x3(tangent, bitangent, normal);

    // 5. Accumulate occlusion
    float occlusion = 0.0;
    for (int i = 0; i < u.sample_count; ++i) {
        // Rotate kernel sample into view space
        float3 sample_dir = mul(kernel[i].xyz, TBN);
        float3 sample_pos = frag_pos + sample_dir * u.radius;

        // Project sample to screen UV
        float4 offset = mul(u.projection, float4(sample_pos, 1.0));
        if (offset.w <= 0.0) continue;
        offset.xyz /= offset.w;
        float2 sample_uv = float2(offset.x * 0.5 + 0.5, 0.5 - offset.y * 0.5);
        if (any(sample_uv < 0.0) || any(sample_uv > 1.0)) continue;

        // Sample actual depth at projected position
        float sample_depth = depth_tex.SampleLevel(depth_sampler, sample_uv, 0).x;
        float3 actual_pos  = reconstruct_view_pos(sample_uv, sample_depth);

        // Occlusion test with range check
        float range_check = 1.0 - smoothstep(0.0, u.radius, abs(frag_pos.z - actual_pos.z));
        occlusion += (actual_pos.z >= sample_pos.z + u.bias ? 1.0 : 0.0) * range_check;
    }

    float ao = 1.0 - (occlusion / float(u.sample_count)) * u.intensity;
    ao = clamp(ao, 0.0, 1.0);
    return float4(ao, 0.0, 0.0, 0.0);
}
