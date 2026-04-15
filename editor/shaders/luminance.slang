struct LuminanceParams {
    uint2 dimensions;
    float adaptation_speed;
    float dt;
    uint has_depth;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct ExposureResult {
    float auto_exposure;
    uint frame_count;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]]
Texture2D<float4> hdr_input;

[[vk::binding(1, 0)]]
SamplerState point_sampler;

[[vk::binding(2, 0)]]
RWStructuredBuffer<ExposureResult> result;

[[vk::binding(3, 0)]]
ConstantBuffer<LuminanceParams> params;

[[vk::binding(4, 0)]]
[NonFiltering] Texture2D<float> depth_input;

static const uint k_group_size = 256;

groupshared float s_sum[k_group_size];
groupshared float s_weight[k_group_size];

[shader("compute")]
[numthreads(k_group_size, 1, 1)]
void cs_main(uint gi : SV_GroupIndex) {
    uint total_pixels = params.dimensions.x * params.dimensions.y;

    float local_sum = 0.0;
    float local_weight = 0.0;

    for (uint i = gi; i < total_pixels; i += k_group_size) {
        uint x = i % params.dimensions.x;
        uint y = i / params.dimensions.x;
        float2 uv = (float2(x, y) + 0.5) / float2(params.dimensions);
        // Skip sky pixels -- depth == 1.0 means nothing was rasterized
        if (params.has_depth != 0u) {
            float depth = depth_input.Load(int3(x, y, 0)).r;
            if (depth >= 1.0) continue;
        }

        float3 color = hdr_input.SampleLevel(point_sampler, uv, 0).rgb;
        float lum = dot(color, float3(0.2126, 0.7152, 0.0722));
        if (lum < 0.0001) continue;

        float log_lum = log2(lum);
        local_sum += clamp(log_lum, -20.0, 20.0);
        local_weight += 1.0;
    }

    s_sum[gi] = local_sum;
    s_weight[gi] = local_weight;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction
    for (uint s = k_group_size / 2; s > 0; s >>= 1) {
        if (gi < s) {
            s_sum[gi] += s_sum[gi + s];
            s_weight[gi] += s_weight[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0 && s_weight[0] > 0.0) {
        float avg_log_lum = s_sum[0] / s_weight[0];

        // Map average luminance to middle gray (0.18)
        float ev = avg_log_lum - log2(0.18);
        float target_exposure = clamp(-ev, -20.0, 20.0);

        float prev_exposure = result[0].auto_exposure;
        uint frame_count = result[0].frame_count;

        float new_exposure;
        if (frame_count == 0u) {
            // First frame: snap directly to target
            new_exposure = target_exposure;
        } else {
            // Temporal smoothing via exponential decay
            float factor = 1.0 - exp(-params.adaptation_speed * params.dt);
            new_exposure = lerp(prev_exposure, target_exposure, factor);
        }

        result[0].auto_exposure = new_exposure;
        result[0].frame_count = frame_count + 1u;
    }
}
