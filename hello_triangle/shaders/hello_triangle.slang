struct Uniforms {
    float4x4 mvp;
    float time;
    float rotation;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> u;

struct VsIn {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float3 color    : COLOR;
};

struct VsOut {
    float4 position : SV_Position;
    float3 color    : COLOR;
};

[shader("vertex")]
VsOut vs_main(VsIn input) {
    // Rotate around Y axis
    float c = cos(u.rotation);
    float s = sin(u.rotation);
    float3 rotated = float3(
        input.position.x * c + input.position.z * s,
        input.position.y,
        -input.position.x * s + input.position.z * c
    );

    VsOut output;
    output.position = mul(u.mvp, float4(rotated, 1.0));

    // Animate color with time-based UV shift
    float phase = u.time * 0.5;
    output.color = input.color * 0.5 + 0.5 * float3(
        sin(phase) * 0.5 + 0.5,
        sin(phase + 2.094) * 0.5 + 0.5,
        sin(phase + 4.189) * 0.5 + 0.5
    );
    return output;
}

[shader("fragment")]
float4 fs_main(float3 color : COLOR) : SV_Target {
    return float4(color, 1.0);
}
