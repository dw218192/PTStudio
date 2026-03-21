static const float PI = 3.14159265;
static const float GGX_MIN_ALPHA = 5e-3;

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

struct Material {
    float3 diffuse_color;
    float metallic;
    float roughness;
    float opacity;
    float2 _pad;
};

struct Light {
    float3 direction_or_pos;
    uint type;
    float3 color;
    float intensity;
    float radius;
    float width;
    float height;
    float angle;  // distant light angular extent in degrees
};

static const uint LIGHT_DISTANT = 0;
static const uint LIGHT_SPHERE  = 1;
static const uint LIGHT_RECT    = 2;
static const uint LIGHT_DISK    = 3;
static const uint LIGHT_DOME    = 4;

// ---------------------------------------------------------------------------
// GGX utilities
// ---------------------------------------------------------------------------

float ggx_alpha(float roughness) {
    return max(roughness * roughness, GGX_MIN_ALPHA);
}

float D_GGX(float NdotH, float a2) {
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float G_SmithGGX1(float cos_theta, float a2) {
    if (cos_theta <= 0.0) return 0.0;
    float cos2 = cos_theta * cos_theta;
    return 2.0 * cos_theta / (cos_theta + sqrt(a2 + (1.0 - a2) * cos2));
}

float G_SmithGGX(float NdotV, float NdotL, float a2) {
    return G_SmithGGX1(NdotV, a2) * G_SmithGGX1(NdotL, a2);
}

float3 F_Schlick(float VdotH, float3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}

// ---------------------------------------------------------------------------
// ShadingGeometry — space-agnostic dot products for BRDF evaluation
// ---------------------------------------------------------------------------

struct ShadingGeometry {
    float NdotL;
    float NdotV;
    float NdotH;
    float VdotH;
    float alpha;
    float a2;
    bool valid;

    static ShadingGeometry from_world(float3 N, float3 L, float3 V, float roughness) {
        ShadingGeometry g;
        g.NdotL = max(dot(N, L), 0.0);
        g.valid = g.NdotL > 0.0;
        float3 H = normalize(L + V);
        g.NdotH = max(dot(N, H), 0.0);
        g.NdotV = max(dot(N, V), 0.001);
        g.VdotH = max(dot(V, H), 0.0);
        g.alpha = max(roughness * roughness, 0.001);
        g.a2 = g.alpha * g.alpha;
        return g;
    }

    static ShadingGeometry from_local(float3 wi, float3 wo, float roughness) {
        ShadingGeometry g;
        g.NdotL = wi.z;
        g.NdotV = wo.z;
        g.valid = wi.z > 0.0 && wo.z > 0.0;
        float3 h = normalize(wi + wo);
        g.NdotH = max(h.z, 0.0);
        g.VdotH = max(dot(wo, h), 0.0);
        g.alpha = max(roughness * roughness, 0.001);
        g.a2 = g.alpha * g.alpha;
        return g;
    }
};

struct TangentFrame {
    float3 T, B, N;

    static TangentFrame build(float3 normal) {
        TangentFrame f;
        f.N = normal;
        float3 up = abs(normal.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
        f.T = normalize(cross(up, normal));
        f.B = cross(normal, f.T);
        return f;
    }

    float3 to_local(float3 w) { return float3(dot(w, T), dot(w, B), dot(w, N)); }
    float3 to_world(float3 w) { return T * w.x + B * w.y + N * w.z; }
};

// ---------------------------------------------------------------------------
// BRDF interface
// ---------------------------------------------------------------------------

interface IBRDF {
    float3 eval(ShadingGeometry g, float3 base_color, float metallic);
    float3 eval_specular(ShadingGeometry g, float3 base_color, float metallic);
    float3 eval_diffuse(ShadingGeometry g, float3 base_color, float metallic);
};

// ---------------------------------------------------------------------------
// Cook-Torrance GGX BRDF
// ---------------------------------------------------------------------------

struct CookTorranceGGX : IBRDF {
    float3 eval(ShadingGeometry g, float3 base_color, float metallic) {
        if (!g.valid) return float3(0.0);
        float3 F0 = lerp(float3(0.04), base_color, metallic);
        float3 F = F_Schlick(g.VdotH, F0);
        float  D = D_GGX(g.NdotH, g.a2);
        float  G = G_SmithGGX(g.NdotV, g.NdotL, g.a2);
        float3 spec = D * G * F / (4.0 * g.NdotV * g.NdotL + 0.001);
        float3 F_nv = F_Schlick(g.NdotV, F0);
        float3 kD = (float3(1.0) - F_nv) * (1.0 - metallic);
        return spec + kD * base_color / PI;
    }

    float3 eval_specular(ShadingGeometry g, float3 base_color, float metallic) {
        if (!g.valid) return float3(0.0);
        float3 F0 = lerp(float3(0.04), base_color, metallic);
        float  D = D_GGX(g.NdotH, g.a2);
        float  G = G_SmithGGX(g.NdotV, g.NdotL, g.a2);
        float3 F = F_Schlick(g.VdotH, F0);
        return D * G * F / (4.0 * g.NdotV * g.NdotL + 0.001);
    }

    float3 eval_diffuse(ShadingGeometry g, float3 base_color, float metallic) {
        float3 F0 = lerp(float3(0.04), base_color, metallic);
        float3 F = F_Schlick(g.NdotV, F0);
        float3 kD = (float3(1.0) - F) * (1.0 - metallic);
        return kD * base_color / PI;
    }
};

// ---------------------------------------------------------------------------
// BRDF sampling (tangent space)
// ---------------------------------------------------------------------------

/// Sample a direction from the combined specular VNDF + cosine hemisphere distribution.
/// wo is in tangent space. Returns wi in tangent space.
float3 sample_brdf(inout Rng rng, float3 wo, float roughness) {
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

// ---------------------------------------------------------------------------
// Light sampling — representative point technique (Karis 2013)
// ---------------------------------------------------------------------------

struct LightSample {
    float3 diffuse_L;           // center direction for diffuse
    float3 spec_L;              // representative direction for specular
    float diffuse_attenuation;  // 1/d^2 from light center
    float spec_attenuation;     // 1/d^2 from representative point
    float spec_roughness;       // roughness widened by light angular extent
    bool valid;
};

LightSample sample_distant_light(Light light, float3 N, float3 V, float roughness) {
    LightSample ls;
    ls.valid = true;
    ls.diffuse_attenuation = 1.0;
    ls.spec_attenuation = 1.0;
    ls.spec_roughness = roughness;

    float3 center_L = -light.direction_or_pos;
    ls.diffuse_L = center_L;

    float half_angle = radians(light.angle * 0.5);

    if (half_angle < 0.001) {
        ls.spec_L = center_L;
        return ls;
    }

    // Representative direction: closest point on the light cone to R
    float3 R = reflect(-V, N);
    float cos_delta = dot(R, center_L);
    float cos_half = cos(half_angle);

    if (cos_delta >= cos_half) {
        ls.spec_L = R;
    } else {
        float3 ortho = R - center_L * cos_delta;
        float ortho_len = length(ortho);
        if (ortho_len > 0.0001) {
            ortho /= ortho_len;
        } else {
            float3 up = abs(center_L.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
            ortho = normalize(cross(center_L, up));
        }
        ls.spec_L = center_L * cos_half + ortho * sin(half_angle);
    }

    // Convolve GGX NDF with the light's angular distribution
    float alpha = roughness * roughness;
    float sin_half = sin(half_angle);
    float alpha_prime = sqrt(alpha * alpha + sin_half * sin_half);
    ls.spec_roughness = sqrt(clamp(alpha_prime, 0.001, 1.0));

    return ls;
}

LightSample sample_area_light(Light light, float3 P, float3 N, float3 V, float roughness) {
    LightSample ls;
    ls.valid = true;
    ls.spec_roughness = roughness;

    float3 center = light.direction_or_pos;
    float effective_radius;

    if (light.type == LIGHT_SPHERE) {
        effective_radius = light.radius;
    } else if (light.type == LIGHT_RECT) {
        effective_radius = sqrt(light.width * light.height) / 2.0;
    } else {
        effective_radius = light.radius;
    }

    float3 to_center = center - P;
    float center_dist = max(length(to_center), 0.001);
    ls.diffuse_L = to_center / center_dist;

    float min_dist2 = effective_radius * effective_radius;
    ls.diffuse_attenuation = 1.0 / max(center_dist * center_dist, min_dist2);

    if (center_dist < effective_radius) {
        ls.valid = false;
        return ls;
    }

    float3 R = reflect(-V, N);
    {
        float center_to_ray = dot(to_center, R);
        float3 closest_on_ray = P + R * max(center_to_ray, 0.0);
        float3 closest_to_center = center - closest_on_ray;
        float closest_dist = length(closest_to_center);

        float3 representative_point;
        if (closest_dist > effective_radius && closest_dist > 0.0001) {
            representative_point = center - normalize(closest_to_center) * effective_radius;
        } else {
            representative_point = closest_on_ray;
        }

        float3 to_rep = representative_point - P;
        float dist = length(to_rep);
        ls.spec_L = to_rep / max(dist, 0.001);
        ls.spec_attenuation = 1.0 / max(dist * dist, min_dist2);

        float alpha = roughness * roughness;
        float sin_half = effective_radius / center_dist;
        float alpha_prime = sqrt(alpha * alpha + sin_half * sin_half);
        ls.spec_roughness = sqrt(clamp(alpha_prime, 0.001, 1.0));
    }

    return ls;
}

LightSample sample_light(Light light, float3 P, float3 N, float3 V, float roughness) {
    if (light.type == LIGHT_DOME) {
        LightSample ls;
        ls.valid = false;
        ls.diffuse_L = float3(0.0);
        ls.spec_L = float3(0.0);
        ls.diffuse_attenuation = 0.0;
        ls.spec_attenuation = 0.0;
        ls.spec_roughness = roughness;
        return ls;
    }

    if (light.type == LIGHT_DISTANT) {
        return sample_distant_light(light, N, V, roughness);
    }

    return sample_area_light(light, P, N, V, roughness);
}

// ---------------------------------------------------------------------------
// Top-level lighting evaluation — generic over BRDF model
// ---------------------------------------------------------------------------

float3 evaluate_lighting<B : IBRDF>(
    B brdf,
    StructuredBuffer<Light> lights, uint light_count,
    float3 P, float3 N, float3 V,
    float3 base_color, float metallic, float roughness,
    out float3 direct_diffuse, out float3 direct_specular
) {
    direct_diffuse = float3(0.0);
    direct_specular = float3(0.0);

    float3 result = float3(0.0);
    for (uint i = 0; i < light_count; i++) {
        Light light = lights[i];

        if (light.type == LIGHT_DOME) {
            result += light.color * light.intensity * base_color * 0.2;
            continue;
        }

        LightSample ls = sample_light(light, P, N, V, roughness);
        if (!ls.valid) continue;

        ShadingGeometry gd = ShadingGeometry::from_world(N, ls.diffuse_L, V, roughness);
        ShadingGeometry gs = ShadingGeometry::from_world(N, ls.spec_L, V, ls.spec_roughness);
        float3 brdf_diffuse = brdf.eval_diffuse(gd, base_color, metallic);
        float3 brdf_specular = brdf.eval_specular(gs, base_color, metallic);

        float3 radiance_diffuse = light.color * light.intensity * ls.diffuse_attenuation;
        float3 radiance_specular = light.color * light.intensity * ls.spec_attenuation;

        float3 diffuse = brdf_diffuse * max(dot(N, ls.diffuse_L), 0.0) * radiance_diffuse;
        float3 specular = brdf_specular * max(dot(N, ls.spec_L), 0.0) * radiance_specular;

        result += diffuse + specular;
        direct_diffuse += diffuse;
        direct_specular += specular;
    }
    return result;
}
