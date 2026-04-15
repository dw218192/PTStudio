// Depth-aware 4x4 bilateral blur. Reads a single-channel source texture (.r)
// and produces a blurred output, rejecting neighbor samples whose depth
// differs from the center by more than `depth_threshold`.

struct BilateralBlurUniforms {
    float2 texel_size;
    float depth_threshold;
    float _pad;
};

[[vk::binding(0, 0)]] ConstantBuffer<BilateralBlurUniforms> u;
[[vk::binding(1, 0)]] Texture2D<float4> input_tex;
[[vk::binding(2, 0)]] [NonFiltering] Texture2D<float>  depth_tex;
[[vk::binding(3, 0)]] SamplerState      linear_sampler;
[[vk::binding(4, 0)]] [NonFiltering]   SamplerState      depth_sampler;

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

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD) : SV_Target0 {
    float center_depth = depth_tex.SampleLevel(depth_sampler, uv, 0).x;
    float center_value = input_tex.SampleLevel(linear_sampler, uv, 0).r;
    float result = center_value;
    float total_weight = 1.0;

    // 4x4 bilateral blur
    for (int y = -2; y <= 1; ++y) {
        for (int x = -2; x <= 1; ++x) {
            float2 offset = float2(float(x) + 0.5, float(y) + 0.5) * u.texel_size;
            float2 sample_uv = uv + offset;

            float v = input_tex.SampleLevel(linear_sampler, sample_uv, 0).r;
            float d = depth_tex.SampleLevel(depth_sampler, sample_uv, 0).x;

            // Bilateral weight: reject samples with large depth discontinuity
            float w = (abs(center_depth - d) < u.depth_threshold) ? 1.0 : 0.0;

            result       += v * w;
            total_weight += w;
        }
    }

    float blurred = (total_weight > 0.0) ? (result / total_weight) : center_value;
    return float4(blurred, blurred, blurred, 1.0);
}
