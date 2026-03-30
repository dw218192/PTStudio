struct DownsampleParams {
    uint output_size;
};

[[vk::binding(0, 0)]] ConstantBuffer<DownsampleParams> params;
[[vk::binding(1, 0)]] Texture2DArray<float4> input_mip;
[[vk::binding(2, 0)]] RWTexture2DArray<float4> output_mip;

[numthreads(8, 8, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    uint face = tid.z;
    if (tid.x >= params.output_size || tid.y >= params.output_size || face >= 6) return;

    // 2x2 box filter: average four source texels
    int2 src = int2(tid.xy) * 2;
    float4 a = input_mip.Load(int4(src.x,     src.y,     face, 0));
    float4 b = input_mip.Load(int4(src.x + 1, src.y,     face, 0));
    float4 c = input_mip.Load(int4(src.x,     src.y + 1, face, 0));
    float4 d = input_mip.Load(int4(src.x + 1, src.y + 1, face, 0));

    output_mip[int3(tid.xy, face)] = (a + b + c + d) * 0.25;
}
