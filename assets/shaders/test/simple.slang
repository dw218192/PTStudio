struct VertexInput {
    float3 position : POSITION;
};

struct VertexOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 1.0);
    return output;
}
