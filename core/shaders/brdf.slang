static const float PI = 3.14159265;
static const float GGX_MIN_ALPHA = 5e-3;

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

float Lambda_GGX(float cos_theta, float a2) {
    if (cos_theta <= 0.0) return 1e10;
    float cos2 = cos_theta * cos_theta;
    return (-1.0 + sqrt(1.0 + a2 * (1.0 - cos2) / cos2)) * 0.5;
}

float G_SmithGGX1(float cos_theta, float a2) {
    if (cos_theta <= 0.0) return 0.0;
    return 1.0 / (1.0 + Lambda_GGX(cos_theta, a2));
}

float G_SmithGGX(float NdotV, float NdotL, float a2) {
    if (NdotV <= 0.0 || NdotL <= 0.0) return 0.0;
    return 1.0 / (1.0 + Lambda_GGX(NdotV, a2) + Lambda_GGX(NdotL, a2));
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
        g.NdotV = max(wo.z, 0.001);
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
