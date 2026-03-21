struct Uniforms {
    float exposure;
    uint mode;   // 0 = ACES, 1 = Reinhard
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> u;

[[vk::binding(1, 0)]]
Texture2D<float4> hdr_input;

[[vk::binding(2, 0)]]
SamplerState linear_sampler;

struct VsOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
};

// Fullscreen triangle from vertex_id (no vertex buffer needed)
[shader("vertex")]
VsOut vs_main(uint vertex_id : SV_VertexID) {
    VsOut output;
    // Generates a triangle that covers [-1,1] clip space
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = float2(uv.x, 1.0 - uv.y); // flip Y for texture sampling
    return output;
}

// ACES filmic tone mapping (Narkowicz 2015 fit)
float3 aces_filmic(float3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Reinhard tone mapping
float3 reinhard(float3 x) {
    return x / (1.0 + x);
}

// Linear to sRGB gamma
float3 linear_to_srgb(float3 c) {
    float3 lo = c * 12.92;
    float3 hi = 1.055 * pow(c, float3(1.0 / 2.4)) - 0.055;
    return select(c <= 0.0031308, lo, hi);
}

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD) : SV_Target0 {
    float3 hdr = hdr_input.Sample(linear_sampler, uv).rgb;

    // Apply exposure
    hdr *= exp2(u.exposure);

    // Tone map
    float3 mapped;
    if (u.mode == 0) {
        mapped = aces_filmic(hdr);
    } else {
        mapped = reinhard(hdr);
    }

    // Linear to sRGB
    mapped = linear_to_srgb(mapped);

    return float4(mapped, 1.0);
}
