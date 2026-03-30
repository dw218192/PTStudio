import ibl_sampling;

struct EquirectParams {
    uint size;
    uint up_axis;  // 1=Y, 2=Z
};

[[vk::binding(0, 0)]] ConstantBuffer<EquirectParams> params;
[[vk::binding(1, 0)]] Texture2D<float4> equirect_tex;
[[vk::binding(2, 0)]] SamplerState equirect_sampler;
[[vk::binding(3, 0)]] RWTexture2DArray<float4> output_cube;

[numthreads(8, 8, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    uint face = tid.z;
    if (tid.x >= params.size || tid.y >= params.size || face >= 6) return;

    float2 uv = (float2(tid.xy) + 0.5) / float(params.size);
    float3 dir = cube_dir_from_face_uv(face, uv);
    if (params.up_axis == 2) {
        dir = float3(dir.x, dir.z, -dir.y);
    }
    float2 eq_uv = equirect_uv_from_dir(dir);

    output_cube[int3(tid.xy, face)] = equirect_tex.SampleLevel(equirect_sampler, eq_uv, 0);
}
