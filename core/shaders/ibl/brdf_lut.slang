import brdf;
import ibl_sampling;

struct BrdfLutParams {
    uint size;
};

[[vk::binding(0, 0)]] ConstantBuffer<BrdfLutParams> params;
[[vk::binding(1, 0)]] RWTexture2D<float4> output_tex;

float geometry_schlick_ibl(float cos_theta, float roughness) {
    float k = (roughness * roughness) / 2.0;
    return cos_theta / (cos_theta * (1.0 - k) + k);
}

float geometry_smith_ibl(float NdotV, float NdotL, float roughness) {
    return geometry_schlick_ibl(NdotV, roughness) * geometry_schlick_ibl(NdotL, roughness);
}

[numthreads(8, 8, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.size || tid.y >= params.size) return;

    float NdotV = max((float(tid.x) + 0.5) / float(params.size), 0.001);
    float roughness = max((float(tid.y) + 0.5) / float(params.size), 0.001);

    float3 V = float3(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);
    float3 N = float3(0, 0, 1);

    float A = 0.0;
    float B = 0.0;

    static const uint SAMPLE_COUNT = 1024;
    for (uint i = 0; i < SAMPLE_COUNT; ++i) {
        float2 Xi = hammersley(i, SAMPLE_COUNT);
        float3 H = importance_sample_ggx(Xi, N, roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if (NdotL > 0.0) {
            float G = geometry_smith_ibl(NdotV, NdotL, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }

    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);

    output_tex[int2(tid.xy)] = float4(A, B, 0.0, 1.0);
}
