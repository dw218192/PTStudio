import lighting;

struct Uniforms {
    float4x4 mvp;
    float3 light_dir;
    float roughness;
    float metallic;
    float scale;
    uint grid_cols;
    uint grid_rows;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> u;

struct VsOut {
    float4 position : SV_Position;
    float magnitude : MAGNITUDE;
};

// Generate hemisphere grid vertex from vertex_id
// Each grid cell = 2 triangles = 6 vertices
// Total vertex count = (grid_cols-1) * (grid_rows-1) * 6
[shader("vertex")]
VsOut vs_main(uint vertex_id : SV_VertexID) {
    VsOut output;

    uint cells_per_row = u.grid_cols - 1;
    uint cells_per_col = u.grid_rows - 1;
    uint cell_id = vertex_id / 6;
    uint vert_in_cell = vertex_id % 6;

    uint cell_col = cell_id % cells_per_row;
    uint cell_row = cell_id / cells_per_row;

    // Two triangles per cell: (0,0)-(1,0)-(0,1) and (1,0)-(1,1)-(0,1)
    uint2 offsets[6] = {
        uint2(0,0), uint2(1,0), uint2(0,1),
        uint2(1,0), uint2(1,1), uint2(0,1)
    };
    uint2 off = offsets[vert_in_cell];
    uint col = cell_col + off.x;
    uint row = cell_row + off.y;

    // Map to spherical coords on upper hemisphere
    float phi = (float(col) / float(u.grid_cols - 1)) * 2.0 * PI;
    float theta = (float(row) / float(u.grid_rows - 1)) * PI * 0.5; // 0 to pi/2

    float3 direction;
    direction.x = sin(theta) * cos(phi);
    direction.y = sin(theta) * sin(phi);
    direction.z = cos(theta);

    // Fixed surface normal pointing up
    float3 N = float3(0, 0, 1);
    float3 V = direction; // outgoing direction (omega_o)
    float3 L = normalize(u.light_dir); // incoming light

    // Evaluate Cook-Torrance specular BRDF
    float3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    float NdotL = max(dot(N, L), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    float alpha = max(u.roughness * u.roughness, 0.001);
    float a2 = alpha * alpha;

    float3 base_color = float3(1.0);
    float3 F0 = lerp(float3(0.04), base_color, u.metallic);

    float D = D_GGX(NdotH, a2);
    float G = G_SmithSchlick(NdotV, NdotL, a2);
    float3 F = F_Schlick(VdotH, F0);

    float spec = D * G * dot(F, float3(1.0/3.0)) / (4.0 * NdotV * NdotL + 0.001);
    float mag = max(spec * NdotL, 0.0);

    // Displace vertex along direction
    float3 pos = direction * (0.1 + mag * u.scale);
    output.position = mul(u.mvp, float4(pos, 1.0));
    output.magnitude = mag;

    return output;
}

// Heat map: blue -> cyan -> green -> yellow -> red
float3 heat_map(float t) {
    t = clamp(t, 0.0, 1.0);
    float3 c;
    if (t < 0.25) {
        c = lerp(float3(0,0,1), float3(0,1,1), t * 4.0);
    } else if (t < 0.5) {
        c = lerp(float3(0,1,1), float3(0,1,0), (t - 0.25) * 4.0);
    } else if (t < 0.75) {
        c = lerp(float3(0,1,0), float3(1,1,0), (t - 0.5) * 4.0);
    } else {
        c = lerp(float3(1,1,0), float3(1,0,0), (t - 0.75) * 4.0);
    }
    return c;
}

[shader("fragment")]
float4 fs_main(float magnitude : MAGNITUDE) : SV_Target0 {
    // Normalize magnitude for heat map — scale so typical range maps to 0..1
    float t = clamp(magnitude * 2.0, 0.0, 1.0);
    float3 color = heat_map(t);
    return float4(color, 1.0);
}
