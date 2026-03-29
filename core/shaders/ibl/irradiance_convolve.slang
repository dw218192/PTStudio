import brdf;
import ibl_sampling;

struct IrradianceParams {
    uint size;
};

[[vk::binding(0, 0)]] ConstantBuffer<IrradianceParams> params;
[[vk::binding(1, 0)]] TextureCube<float4> env_cubemap;
[[vk::binding(2, 0)]] SamplerState env_sampler;
[[vk::binding(3, 0)]] RWTexture2DArray<float4> output_irradiance;

[numthreads(8, 8, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    uint face = tid.z;
    if (tid.x >= params.size || tid.y >= params.size || face >= 6) return;

    float2 uv = (float2(tid.xy) + 0.5) / float(params.size);
    float3 N = cube_dir_from_face_uv(face, uv);

    float3 up = abs(N.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 right = normalize(cross(up, N));
    up = cross(N, right);

    float3 irradiance = float3(0, 0, 0);

    static const uint SAMPLE_COUNT = 128;
    for (uint i = 0; i < SAMPLE_COUNT; ++i) {
        float2 Xi = hammersley(i, SAMPLE_COUNT);

        // Cosine-weighted hemisphere sampling
        float cos_theta = sqrt(1.0 - Xi.y);
        float sin_theta = sqrt(Xi.y);
        float phi = 2.0 * PI * Xi.x;

        float3 local_dir = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
        float3 sample_dir = right * local_dir.x + up * local_dir.y + N * local_dir.z;

        irradiance += env_cubemap.SampleLevel(env_sampler, sample_dir, 0).rgb;
    }

    // Cosine-weighted: integral = PI * (1/N) * sum(L)
    irradiance = PI * irradiance / float(SAMPLE_COUNT);
    output_irradiance[int3(tid.xy, face)] = float4(irradiance, 1.0);
}
