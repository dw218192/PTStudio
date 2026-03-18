import lighting;

struct Uniforms {
    float4x4 mvp;
    float3 light_dir;
    float roughness;
    float metallic;
    float scale;
    uint grid_cols;
    uint grid_rows;
    uint mode;       // 0 = specular, 1 = diffuse
    uint _pad;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> u;

struct VsOut {
    float4 position : SV_Position;
    float magnitude : MAGNITUDE;
    uint mode : MODE;
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
    float3 base_color = float3(1.0);

    CookTorranceGGX brdf;
    float mag;
    if (u.mode == 0) {
        float3 spec = brdf.evaluate_specular(N, L, V, base_color, u.metallic, u.roughness);
        mag = length(spec);
    } else {
        float3 diff = brdf.evaluate_diffuse(N, L, V, base_color, u.metallic);
        mag = length(diff);
    }

    // Displace vertex along direction
    float3 pos = direction * (0.1 + mag * u.scale);
    output.position = mul(u.mvp, float4(pos, 1.0));
    output.magnitude = mag;
    output.mode = u.mode;

    return output;
}

// Warm heat map: blue -> cyan -> green -> yellow -> red (specular)
float3 warm_heat_map(float t) {
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

// Cool color map: purple -> blue -> cyan (diffuse)
float3 cool_color_map(float t) {
    t = clamp(t, 0.0, 1.0);
    float3 c;
    if (t < 0.5) {
        c = lerp(float3(0.4, 0.0, 0.8), float3(0.0, 0.2, 1.0), t * 2.0);
    } else {
        c = lerp(float3(0.0, 0.2, 1.0), float3(0.0, 0.8, 1.0), (t - 0.5) * 2.0);
    }
    return c;
}

[shader("fragment")]
float4 fs_main(float magnitude : MAGNITUDE, uint mode : MODE) : SV_Target0 {
    float t = clamp(magnitude * 2.0, 0.0, 1.0);
    float3 color;
    if (mode == 0) {
        color = warm_heat_map(t);
    } else {
        color = cool_color_map(t);
    }
    return float4(color, 1.0);
}
