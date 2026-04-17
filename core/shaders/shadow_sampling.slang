/// Per-light shadow info (must match C++ ShadowInfo, 96 bytes).
/// One entry per light. Lights without shadows have has_shadow == 0.
public struct ShadowInfo {
    public float4x4 light_vp;        // 64 bytes
    public float texel_size;         //  4 bytes
    public float normal_bias;        //  4 bytes
    public uint has_shadow;          //  4 bytes -- 0 = inactive, 1 = active
    public uint layer;               //  4 bytes -- shadow map array layer
    public float light_near;         //  4 bytes -- light-space near plane
    public float light_far;          //  4 bytes -- light-space far plane
    public float light_size_uv;      //  4 bytes -- PCSS light size (see below)
    public uint projection_type;     //  4 bytes -- 0 = ortho, 1 = perspective
};

// Non-PCSS baseline: 5x5 PCF with manual depth comparison.
public float compute_shadow_pcf(
    Texture2DArray<float> shadow_map,
    SamplerState shadow_sampler,
    float3 shadow_coord,    // xy = UV in [0,1], z = reference depth in [0,1]
    uint layer,
    float texel_size        // 1.0 / shadow_map_resolution
) {
    float shadow = 0.0;
    float ref_depth = shadow_coord.z;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            float2 offset = float2(float(x), float(y)) * texel_size;
            float2 uv = shadow_coord.xy + offset;
            float sampled_depth = shadow_map.SampleLevel(
                shadow_sampler,
                float3(uv, float(layer)),
                0
            ).x;
            shadow += (ref_depth <= sampled_depth) ? 1.0 : 0.0;
        }
    }
    return shadow / 25.0;
}

// ---------------------------------------------------------------------------
// PCSS (Percentage-Closer Soft Shadows)
//
// Semantics of `light_size_uv` (computed CPU-side in shadowMapPass):
//   * Ortho (distant): tan(half_angle) / ortho_width. Penumbra formula:
//       penumbra_uv = light_size_uv * (d_r - d_b) / d_b
//   * Perspective (rect/disk): light_radius / (2 * tan(fov/2)).
//     Penumbra formula (depth-dependent because UV scale shrinks with depth):
//       penumbra_uv = light_size_uv * (d_r - d_b) / (d_b * d_r)
//
// Both d_r (receiver) and d_b (blocker) are LIGHT-SPACE LINEAR depths
// reconstructed from the shadow map's stored NDC depth via `linearize_depth`.
// ---------------------------------------------------------------------------

// Shared knobs
public static const uint k_pcss_blocker_taps = 16;
public static const uint k_pcss_pcf_taps = 16;
public static const float k_pcss_min_penumbra_uv = 1e-5;

// 16-sample Poisson disk in [-1, 1]^2.
public static const float2 k_poisson_disk_16[16] = {
    float2(-0.94201624,  -0.39906216),
    float2( 0.94558609,  -0.76890725),
    float2(-0.09418410,  -0.92938870),
    float2( 0.34495938,   0.29387760),
    float2(-0.91588581,   0.45771432),
    float2(-0.81544232,  -0.87912464),
    float2(-0.38277543,   0.27676845),
    float2( 0.97484398,   0.75648379),
    float2( 0.44323325,  -0.97511554),
    float2( 0.53742981,  -0.47373420),
    float2(-0.26496911,  -0.41893023),
    float2( 0.79197514,   0.19090188),
    float2(-0.24188840,   0.99706507),
    float2(-0.81409955,   0.91437590),
    float2( 0.19984126,   0.78641367),
    float2( 0.14383161,  -0.14100790)
};

// Reconstruct light-space linear depth from NDC z in [0,1].
// Matches glm::ortho / glm::perspective with GLM_FORCE_DEPTH_ZERO_TO_ONE.
public float linearize_depth(float ndc_z, float near_plane, float far_plane, uint projection_type) {
    if (projection_type == 0u) {
        // Ortho: ndc_z = (view_dist - near) / (far - near)
        return near_plane + ndc_z * (far_plane - near_plane);
    }
    // Perspective: ndc_z = far/(far-near) - (far*near)/((far-near)*view_dist)
    float denom = far_plane - ndc_z * (far_plane - near_plane);
    return (near_plane * far_plane) / max(denom, 1e-6);
}

// Simple interleaved gradient noise for per-pixel rotation of the Poisson disk.
internal float pcss_noise(float2 frag_uv) {
    return frac(sin(dot(frag_uv, float2(12.9898, 78.233))) * 43758.5453);
}

internal float2x2 pcss_rotation(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return float2x2(c, -s, s, c);
}

/// Blocker search: returns average blocker linear depth, or -1 if no blockers.
///
/// `receiver_ndc` is the receiver's post-projection depth in [0,1]. Because
/// NDC depth is monotonic in light-space linear depth under both ortho and
/// perspective projections, the "is this tap closer to the light?" test can
/// be done directly in NDC -- no linearize per tap. We only linearize the
/// taps that classify as blockers, then average in linear space (required
/// for the similar-triangles penumbra formula).
public float pcss_blocker_search(
    Texture2DArray<float> shadow_map,
    SamplerState shadow_sampler,
    float2 shadow_uv,
    uint layer,
    float receiver_ndc,
    float search_radius_uv,
    float near_plane,
    float far_plane,
    uint projection_type,
    float2 rotation_seed
) {
    float2x2 rot = pcss_rotation(pcss_noise(rotation_seed) * 6.2831853);
    float total = 0.0;
    float count = 0.0;
    for (uint i = 0; i < k_pcss_blocker_taps; ++i) {
        float2 offset = mul(rot, k_poisson_disk_16[i]) * search_radius_uv;
        float sampled_ndc = shadow_map.SampleLevel(
            shadow_sampler, float3(shadow_uv + offset, float(layer)), 0).x;
        // Classify in NDC (monotonic). Linearize only on hit.
        if (sampled_ndc < receiver_ndc) {
            total += linearize_depth(sampled_ndc, near_plane, far_plane, projection_type);
            count += 1.0;
        }
    }
    return (count > 0.0) ? (total / count) : -1.0;
}

/// Penumbra size in UV space. Receiver/blocker depths are LIGHT-SPACE LINEAR.
public float pcss_penumbra_uv(float receiver_lin, float blocker_lin, float light_size_uv,
                              uint projection_type) {
    float delta = max(receiver_lin - blocker_lin, 0.0);
    float denom = max(blocker_lin, 1e-6);
    if (projection_type == 0u) {
        // Ortho (distant light): penumbra_uv = light_size_uv * (d_r - d_b) / d_b
        return light_size_uv * delta / denom;
    }
    // Perspective (area light): also divide by d_r to account for UV scale
    // shrinking with receiver depth.
    return light_size_uv * delta / (denom * max(receiver_lin, 1e-6));
}

/// Variable-kernel PCF: sample_count taps in a disk of size kernel_uv.
public float pcss_pcf_variable(
    Texture2DArray<float> shadow_map,
    SamplerState shadow_sampler,
    float2 shadow_uv,
    uint layer,
    float ref_ndc_depth,
    float kernel_uv,
    float2 rotation_seed
) {
    float2x2 rot = pcss_rotation(pcss_noise(rotation_seed) * 6.2831853);
    float shadow = 0.0;
    for (uint i = 0; i < k_pcss_pcf_taps; ++i) {
        float2 offset = mul(rot, k_poisson_disk_16[i]) * kernel_uv;
        float sampled = shadow_map.SampleLevel(
            shadow_sampler, float3(shadow_uv + offset, float(layer)), 0).x;
        shadow += (ref_ndc_depth <= sampled) ? 1.0 : 0.0;
    }
    return shadow / float(k_pcss_pcf_taps);
}

/// Full PCSS: blocker search -> penumbra estimation -> variable PCF.
/// Falls back to the fixed PCF kernel when `light_size_uv <= 0` (no defined
/// physical light size, e.g. a delta distant light or a legacy code path).
public float compute_shadow_pcss(
    Texture2DArray<float> shadow_map,
    SamplerState shadow_sampler,
    float3 shadow_coord,
    ShadowInfo si,
    float2 rotation_seed
) {
    if (si.light_size_uv <= 0.0) {
        return compute_shadow_pcf(shadow_map, shadow_sampler, shadow_coord, si.layer, si.texel_size);
    }

    float receiver_lin = linearize_depth(
        shadow_coord.z, si.light_near, si.light_far, si.projection_type);

    // Blocker search kernel: the portion of the light that can cast a shadow
    // onto the receiver. We use a conservative fixed fraction of the light
    // size, which is sufficient for our depth-range cases.
    float search_radius_uv = max(si.light_size_uv * 0.5, si.texel_size * 2.0);

    float blocker_lin = pcss_blocker_search(
        shadow_map, shadow_sampler, shadow_coord.xy, si.layer, shadow_coord.z,
        search_radius_uv, si.light_near, si.light_far, si.projection_type, rotation_seed);
    if (blocker_lin < 0.0) {
        return 1.0;  // no blockers -> fully lit
    }

    float penumbra_uv = pcss_penumbra_uv(
        receiver_lin, blocker_lin, si.light_size_uv, si.projection_type);
    // Clamp to at least 1 texel so we don't degenerate to a single tap.
    penumbra_uv = max(penumbra_uv, si.texel_size);

    return pcss_pcf_variable(
        shadow_map, shadow_sampler, shadow_coord.xy, si.layer, shadow_coord.z,
        penumbra_uv, rotation_seed);
}
