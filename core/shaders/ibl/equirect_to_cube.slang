// Equirectangular HDR -> cubemap face conversion.
// Dispatched once per face (Z=1) with a single-layer output view.
// Per-face dispatch works around a Dawn/Tint D3D12 codegen issue where
// textureStore to texture_storage_2d_array silently drops writes to
// array layers > 0 (Slang emits mixed u32/i32 textureStore coordinates
// that some HLSL compilers mishandle).
import ibl_sampling;

struct EquirectParams {
    uint size;
    uint up_axis;  // 1=Y, 2=Z
    uint face;     // cubemap face index (0-5), set per dispatch
    uint _pad;
};

[[vk::binding(0, 0)]] ConstantBuffer<EquirectParams> params;
[[vk::binding(1, 0)]] Texture2D<float4> equirect_tex;
[[vk::binding(2, 0)]] SamplerState equirect_sampler;
[[vk::binding(3, 0)]] RWTexture2DArray<float4> output_cube;

[numthreads(8, 8, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.size || tid.y >= params.size) return;

    float2 uv = (float2(tid.xy) + 0.5) / float(params.size);
    float3 dir = cube_dir_from_face_uv(params.face, uv);
    if (params.up_axis == 2) {
        dir = float3(dir.x, dir.z, -dir.y);
    }
    float2 eq_uv = equirect_uv_from_dir(dir);

    output_cube[int3(tid.xy, 0)] = equirect_tex.SampleLevel(equirect_sampler, eq_uv, 0);
}
