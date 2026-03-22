struct Uniforms {
    float4x4 inv_vp;
    float4x4 vp;
    float3 camera_pos;
    float near_plane;
    float far_plane;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> u;

struct VsOut {
    float4 position   : SV_Position;
    float3 near_point : NEAR_POINT;
    float3 far_point  : FAR_POINT;
};

// Full-screen triangle from vertex IDs — no vertex buffer needed.
[shader("vertex")]
VsOut vs_main(uint vid : SV_VertexID) {
    float2 uv  = float2((vid << 1) & 2, vid & 2);
    float2 ndc = uv * 2.0 - 1.0;

    // Unproject near / far points into world space
    float4 near_h = mul(u.inv_vp, float4(ndc, 0.0, 1.0));
    float4 far_h  = mul(u.inv_vp, float4(ndc, 1.0, 1.0));

    VsOut o;
    o.position   = float4(ndc, 0.0, 1.0);
    o.near_point = near_h.xyz / near_h.w;
    o.far_point  = far_h.xyz / far_h.w;
    return o;
}

// ── helpers ──────────────────────────────────────────────────────────

float grid_line(float2 coord, float scale) {
    float2 scaled   = coord * scale;
    float2 deriv    = fwidth(scaled);
    float2 grid     = abs(fract(scaled - 0.5) - 0.5) / deriv;
    return 1.0 - min(min(grid.x, grid.y), 1.0);
}

float compute_depth(float3 world_pos) {
    float4 clip = mul(u.vp, float4(world_pos, 1.0));
    return clip.z / clip.w;
}

// ── fragment ─────────────────────────────────────────────────────────

struct FsOut {
    float4 color : SV_Target;
    float  depth : SV_Depth;
};

[shader("fragment")]
FsOut fs_main(float3 near_point : NEAR_POINT, float3 far_point : FAR_POINT) {
    FsOut o;

    // Ray–plane intersection with Y = 0
    float denom = far_point.y - near_point.y;
    float t     = -near_point.y / denom;
    float3 world_pos = near_point + t * (far_point - near_point);

    // Clip-space depth for the intersection point
    float depth = compute_depth(world_pos);

    // Discard fragments behind camera or outside clip volume
    if (t < 0.0 || depth < 0.0 || depth > 1.0) discard;

    // ── distance fade ────────────────────────────────────────────────
    float dist = length(world_pos.xz - u.camera_pos.xz);
    float fade = 1.0 - smoothstep(20.0, 150.0, dist);
    if (fade <= 0.0) discard;

    // ── grid lines (1-unit and 10-unit) ──────────────────────────────
    float2 coord = world_pos.xz;
    float line1  = grid_line(coord, 1.0);
    float line10 = grid_line(coord, 0.1);

    float alpha  = max(line1 * 0.3, line10 * 0.5) * fade;
    float3 color = float3(0.5, 0.5, 0.5);

    // ── axis highlighting ────────────────────────────────────────────
    float2 deriv = fwidth(coord);

    // X axis (Z = 0) -> red, darker for negative X
    if (abs(coord.y) < deriv.y * 1.5) {
        color = float3(0.8, 0.15, 0.15);
        if (coord.x < 0) {
            color *= 0.2;
        }
        alpha = max(alpha, fade * 0.7);
    }

    // Z axis (X = 0) -> blue, darker for negative Z
    if (abs(coord.x) < deriv.x * 1.5) {
        color = float3(0.15, 0.15, 0.8);
        if (coord.y < 0) {
            color *= 0.2;
        }
        alpha = max(alpha, fade * 0.7);
    }

    if (alpha < 0.001) discard;

    // Premultiplied alpha output
    o.color = float4(color * alpha, alpha);
    // Push grid slightly behind coplanar scene geometry to avoid z-fighting
    o.depth = depth + 0.0001;
    return o;
}
