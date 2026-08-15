struct BlitUniforms { uint width; uint height; uint _p0; uint _p1; };

[[vk::binding(0, 0)]] ConstantBuffer<BlitUniforms> u;
[[vk::binding(1, 0)]] StructuredBuffer<float4> buf;

[shader("vertex")]
float4 vs_main(uint vid : SV_VertexID) : SV_Position {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}

[shader("fragment")]
float4 fs_main(float4 pos : SV_Position) : SV_Target0 {
    uint px = uint(pos.x);
    uint py = uint(pos.y);
    return buf[py * u.width + px];
}
