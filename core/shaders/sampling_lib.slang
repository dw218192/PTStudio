import brdf_lib;

// ---------------------------------------------------------------------------
// RNG (PCG hash)
// ---------------------------------------------------------------------------

struct Rng {
    uint state;

    static Rng create(uint seed) { Rng r; r.state = seed; return r; }

    [mutating] float next() {
        state = state * 747796405u + 2891336453u;
        uint w = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return float((w >> 22u) ^ w) / 4294967295.0;
    }
};

// ---------------------------------------------------------------------------
// BRDF sampling (tangent space)
// ---------------------------------------------------------------------------

/// Sample a direction from the combined specular VNDF + cosine hemisphere distribution.
/// wo is in tangent space. Returns wi in tangent space.
float3 sample_brdf(inout Rng rng, float3 wo, float roughness) {
    if (wo.z <= 0.0) return float3(0, 0, -1);
    if (rng.next() < 0.5) {
        // GGX VNDF (Heitz 2018)
        float a = ggx_alpha(roughness);
        float3 wh = normalize(float3(a * wo.x, a * wo.y, wo.z));
        if (wh.z < 0.0) wh = -wh;
        float3 T1 = (wh.z < 0.99999) ? normalize(cross(float3(0, 0, 1), wh)) : float3(1, 0, 0);
        float3 T2 = cross(wh, T1);
        // Concentric disk sample
        float u1 = rng.next() * 2.0 - 1.0;
        float u2 = rng.next() * 2.0 - 1.0;
        float disk_r, disk_theta;
        if (u1 == 0.0 && u2 == 0.0) { disk_r = 0.0; disk_theta = 0.0; }
        else if (abs(u1) > abs(u2)) { disk_r = u1; disk_theta = (PI / 4.0) * (u2 / u1); }
        else { disk_r = u2; disk_theta = (PI / 2.0) - (PI / 4.0) * (u1 / u2); }
        float dx = disk_r * cos(disk_theta);
        float dy = disk_r * sin(disk_theta);
        float h = sqrt(1.0 - dx * dx);
        float y = lerp(h, dy, (1.0 + wh.z) / 2.0);
        float3 nh = dx * T1 + y * T2 + sqrt(max(0.0, 1.0 - dx * dx - y * y)) * wh;
        nh = normalize(float3(nh.x * a, nh.y * a, max(1e-6, nh.z)));
        return 2.0 * dot(wo, nh) * nh - wo;
    } else {
        // Cosine hemisphere
        float s1 = rng.next(), s2 = rng.next();
        float cos_t = sqrt(1.0 - s2), sin_t = sqrt(s2);
        float phi = 2.0 * PI * s1;
        return float3(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
    }
}

/// PDF of sample_brdf for a given wi/wo pair in tangent space.
float pdf_brdf(float3 wi, float3 wo, float roughness) {
    if (wi.z <= 0.0 || wo.z <= 0.0) return 0.0;
    float3 h = normalize(wi + wo);
    float a = ggx_alpha(roughness);
    float a2 = a * a;
    float denom = 4.0 * abs(wo.z);
    float pdf_spec = (denom < 0.0001) ? 0.0 : G_SmithGGX1(wo.z, a2) * D_GGX(h.z, a2) / denom;
    float pdf_diff = wi.z / PI;
    return 0.5 * pdf_spec + 0.5 * pdf_diff;
}
