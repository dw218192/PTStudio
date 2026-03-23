// LTC (Linearly Transformed Cosines) area light evaluation.
// Reference: Heitz et al., "Real-Time Polygonal-Light Shading with LTC", SIGGRAPH 2016

import brdf;
import light;

// ---------------------------------------------------------------------------
// LTC utilities
// ---------------------------------------------------------------------------

/// Compute LUT UV coordinates from NdotV and roughness.
/// Convention matches Heitz reference: U = roughness, V = sqrt(1 - NdotV).
float2 ltc_coords(float NdotV, float roughness) {
    float2 uv = float2(roughness, sqrt(1.0 - clamp(NdotV, 0.0, 1.0)));
    // Half-texel offset for correct sampling at texel centers
    uv = uv * (63.0 / 64.0) + (0.5 / 64.0);
    return uv;
}

/// Reconstruct 3x3 M^(-1) matrix from 4 stored parameters.
/// The matrix has structure: | a  0  b |
///                           | 0  1  0 |
///                           | c  0  d |
float3x3 ltc_matrix(float4 t) {
    return float3x3(
        float3(t.x, 0, t.y),
        float3(  0, 1,   0),
        float3(t.z, 0, t.w)
    );
}

/// Single edge contribution to the polygon form factor integral.
/// v1, v2 must be on the unit sphere (normalized).
float integrate_edge(float3 v1, float3 v2) {
    float cos_theta = clamp(dot(v1, v2), -0.9999, 0.9999);
    float theta = acos(cos_theta);
    float3 c = cross(v1, v2);
    float len = length(c);
    if (len < 1e-7) return 0.0;
    return theta * (c.z / len);
}

/// Clip a polygon edge against the z=0 horizon plane.
/// Returns the number of output vertices written (0, 1, or 2).
/// Vertices with z < 0 are clipped to the horizon.
int clip_edge(float3 a, float3 b, out float3 o1, out float3 o2) {
    o1 = float3(0);
    o2 = float3(0);

    if (a.z >= 0.0 && b.z >= 0.0) {
        o1 = a;
        o2 = b;
        return 2;
    }
    if (a.z < 0.0 && b.z < 0.0) {
        return 0;
    }
    // One above, one below — interpolate to find horizon crossing
    float t = a.z / (a.z - b.z);
    float3 mid = a + t * (b - a);
    if (a.z >= 0.0) {
        o1 = a;
        o2 = mid;
        return 2;
    } else {
        o1 = mid;
        o2 = b;
        return 2;
    }
}

// ---------------------------------------------------------------------------
// LTC polygon evaluation (up to 8 vertices after clipping)
// ---------------------------------------------------------------------------

/// Evaluate the clamped cosine integral over a polygon, with horizon clipping.
/// Input: polygon vertices in tangent space (z = up = normal direction).
/// Up to 4 input vertices (rect) or 16 (disk N-gon).
/// Returns the integral value (always >= 0).
float ltc_evaluate_clipped_quad(float3 v0, float3 v1, float3 v2, float3 v3) {
    // Clip against horizon (z >= 0) and evaluate integral
    // For simplicity, we clamp z to a small positive epsilon and re-normalize.
    // This avoids full polygon clipping complexity while being visually correct
    // for most configurations.
    float3 verts[4] = { v0, v1, v2, v3 };
    for (int i = 0; i < 4; i++) {
        verts[i].z = max(verts[i].z, 1e-5);
        verts[i] = normalize(verts[i]);
    }

    float sum = 0.0;
    sum += integrate_edge(verts[0], verts[1]);
    sum += integrate_edge(verts[1], verts[2]);
    sum += integrate_edge(verts[2], verts[3]);
    sum += integrate_edge(verts[3], verts[0]);

    return max(0.0, sum / (2.0 * PI));
}

// ---------------------------------------------------------------------------
// Rect light LTC evaluation
// ---------------------------------------------------------------------------

void ltc_evaluate_rect(
    Light light, float3 P, float3 N, float3 V,
    float roughness, float3 base_color, float metallic,
    Texture2D<float4> ltc_mat_tex, Texture2D<float2> ltc_amp_tex,
    SamplerState ltc_samp,
    out float3 out_diffuse, out float3 out_specular
) {
    out_diffuse = float3(0.0);
    out_specular = float3(0.0);

    float3 center = light.direction_or_pos;
    if (!is_front_facing(P, center, light.right, light.up)) return;

    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    float2 uv = ltc_coords(NdotV, roughness);

    // Sample LUTs
    float4 t1 = ltc_mat_tex.SampleLevel(ltc_samp, uv, 0);
    float2 t2 = ltc_amp_tex.SampleLevel(ltc_samp, uv, 0);
    float3x3 Minv = ltc_matrix(t1);

    // Build view-aligned tangent frame (required by LTC fitting convention).
    // T1 = projection of V onto tangent plane, T2 = cross(N, T1).
    float3 T1 = normalize(V - N * dot(V, N));
    float3 T2 = cross(N, T1);

    // Compute 4 rect corners in world space
    float3 c0 = center - light.right - light.up;
    float3 c1 = center + light.right - light.up;
    float3 c2 = center + light.right + light.up;
    float3 c3 = center - light.right + light.up;

    // Transform to tangent space (T1, T2, N), relative to shading point
    float3 d0 = c0 - P, d1 = c1 - P, d2 = c2 - P, d3 = c3 - P;
    float3 L0 = float3(dot(d0, T1), dot(d0, T2), dot(d0, N));
    float3 L1 = float3(dot(d1, T1), dot(d1, T2), dot(d1, N));
    float3 L2 = float3(dot(d2, T1), dot(d2, T2), dot(d2, N));
    float3 L3 = float3(dot(d3, T1), dot(d3, T2), dot(d3, N));

    // DIFFUSE: integrate with identity M (clamped cosine over polygon)
    float diff_integral = ltc_evaluate_clipped_quad(
        normalize(L0), normalize(L1), normalize(L2), normalize(L3));

    // SPECULAR: apply M^(-1) and integrate
    float spec_integral = ltc_evaluate_clipped_quad(
        normalize(mul(Minv, L0)),
        normalize(mul(Minv, L1)),
        normalize(mul(Minv, L2)),
        normalize(mul(Minv, L3)));

    // Apply BRDF scaling
    float3 F0 = lerp(float3(0.04), base_color, metallic);
    float3 spec_color = F0 * t2.x + (1.0 - F0) * t2.y;
    float3 kD = (1.0 - spec_color) * (1.0 - metallic);

    float3 radiance = light.color * light.intensity;
    out_specular = spec_color * spec_integral * radiance;
    out_diffuse = kD * base_color * diff_integral * radiance;
}

// ---------------------------------------------------------------------------
// Disk light LTC evaluation (approximate as N-gon)
// ---------------------------------------------------------------------------

void ltc_evaluate_disk(
    Light light, float3 P, float3 N, float3 V,
    float roughness, float3 base_color, float metallic,
    Texture2D<float4> ltc_mat_tex, Texture2D<float2> ltc_amp_tex,
    SamplerState ltc_samp,
    out float3 out_diffuse, out float3 out_specular
) {
    out_diffuse = float3(0.0);
    out_specular = float3(0.0);

    float3 center = light.direction_or_pos;
    if (!is_front_facing(P, center, light.right, light.up)) return;

    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    float2 uv = ltc_coords(NdotV, roughness);

    float4 t1 = ltc_mat_tex.SampleLevel(ltc_samp, uv, 0);
    float2 t2 = ltc_amp_tex.SampleLevel(ltc_samp, uv, 0);
    float3x3 Minv = ltc_matrix(t1);

    // View-aligned tangent frame
    float3 T1 = normalize(V - N * dot(V, N));
    float3 T2 = cross(N, T1);

    // Generate 16-gon vertices on the disk circumference
    static const int NSEG = 16;
    float3 disk_local[NSEG];
    float3 disk_spec[NSEG];
    for (int i = 0; i < NSEG; i++) {
        float a = 2.0 * PI * float(i) / float(NSEG);
        float3 world_pt = center + light.right * cos(a) + light.up * sin(a);
        float3 d = world_pt - P;
        float3 L = float3(dot(d, T1), dot(d, T2), dot(d, N));
        L.z = max(L.z, 1e-5);
        disk_local[i] = normalize(L);
        disk_spec[i] = normalize(mul(Minv, L));
    }

    // Evaluate polygon integrals
    float diff_sum = 0.0;
    float spec_sum = 0.0;
    for (int i = 0; i < NSEG; i++) {
        int j = (i + 1) % NSEG;
        diff_sum += integrate_edge(disk_local[i], disk_local[j]);
        spec_sum += integrate_edge(disk_spec[i], disk_spec[j]);
    }
    float diff_integral = max(0.0, diff_sum / (2.0 * PI));
    float spec_integral = max(0.0, spec_sum / (2.0 * PI));

    float3 F0 = lerp(float3(0.04), base_color, metallic);
    float3 spec_color = F0 * t2.x + (1.0 - F0) * t2.y;
    float3 kD = (1.0 - spec_color) * (1.0 - metallic);

    float3 radiance = light.color * light.intensity;
    out_specular = spec_color * spec_integral * radiance;
    out_diffuse = kD * base_color * diff_integral * radiance;
}
