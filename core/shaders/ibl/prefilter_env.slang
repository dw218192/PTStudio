import brdf;
import ibl_sampling;

struct PrefilterParams {
    uint size;
    float roughness;
    uint env_size;
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

    float a = params.roughness * params.roughness;
    float a2 = a * a;
    float sa_texel = 4.0 * PI / (6.0 * float(params.env_size) * float(params.env_size));

    static const uint SAMPLE_COUNT = 512;
    for (uint i = 0; i < SAMPLE_COUNT; ++i) {
        float2 Xi = hammersley(i, SAMPLE_COUNT);
        float3 H = importance_sample_ggx(Xi, N, params.roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.001);
            float D = D_GGX(NdotH, a2);
            float pdf = D * NdotH / (4.0 * VdotH);
            float sa_sample = 1.0 / (float(SAMPLE_COUNT) * max(pdf, 0.001));
            float mip = params.roughness == 0.0 ? 0.0 : 0.5 * log2(sa_sample / sa_texel);

            prefiltered += env_cubemap.SampleLevel(env_sampler, L, max(mip, 0.0)).rgb * NdotL;
            total_weight += NdotL;
        }
    }

    prefiltered /= max(total_weight, 0.001);
    output_prefiltered[int3(tid.xy, face)] = float4(prefiltered, 1.0);
}
