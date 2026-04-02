// Box-filter downsample for cubemap mipchain.
// Dispatched once per face with a single-layer output view (see equirect_to_cube.slang).
struct DownsampleParams {
    uint output_size;
    uint face;  // cubemap face index (0-5), set per dispatch
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] ConstantBuffer<DownsampleParams> params;
[[vk::binding(1, 0)]] Texture2DArray<float4> input_mip;
[[vk::binding(2, 0)]] RWTexture2DArray<float4> output_mip;

[numthreads(8, 8, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.output_size || tid.y >= params.output_size) return;

    // 2x2 box filter: average four source texels
    int2 src = int2(tid.xy) * 2;
    float4 a = input_mip.Load(int4(src.x,     src.y,     params.face, 0));
    float4 b = input_mip.Load(int4(src.x + 1, src.y,     params.face, 0));
    float4 c = input_mip.Load(int4(src.x,     src.y + 1, params.face, 0));
    float4 d = input_mip.Load(int4(src.x + 1, src.y + 1, params.face, 0));

    output_mip[int3(tid.xy, 0)] = (a + b + c + d) * 0.25;
}
