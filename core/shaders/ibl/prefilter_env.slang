import brdf;
import ibl_sampling;

struct PrefilterParams {
    uint size;
    float roughness;
};

[[vk::binding(0, 0)]] ConstantBuffer<PrefilterParams> params;
[[vk::binding(1, 0)]] TextureCube<float4> env_cubemap;
[[vk::binding(2, 0)]] SamplerState env_sampler;
[[vk::binding(3, 0)]] RWTexture2DArray<float4> output_prefiltered;

[numthreads(8, 8, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    uint face = tid.z;
    if (tid.x >= params.size || tid.y >= params.size || face >= 6) return;

    float2 uv = (float2(tid.xy) + 0.5) / float(params.size);
    float3 N = cube_dir_from_face_uv(face, uv);
    float3 R = N;
    float3 V = R;

    float3 prefiltered = float3(0, 0, 0);
    float total_weight = 0.0;

    static const uint SAMPLE_COUNT = 512;
    for (uint i = 0; i < SAMPLE_COUNT; ++i) {
        float2 Xi = hammersley(i, SAMPLE_COUNT);
        float3 H = importance_sample_ggx(Xi, N, params.roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            prefiltered += env_cubemap.SampleLevel(env_sampler, L, 0).rgb * NdotL;
            total_weight += NdotL;
        }
    }

    prefiltered /= max(total_weight, 0.001);
    output_prefiltered[int3(tid.xy, face)] = float4(prefiltered, 1.0);
}
