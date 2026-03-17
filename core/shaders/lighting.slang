static const float PI = 3.14159265;

struct Material {
    float3 diffuse_color;
    float metallic;
    float roughness;
    float opacity;
    float2 _pad;
};

struct GpuLight {
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
// BRDF interface — implement to swap shading models at compile time
// ---------------------------------------------------------------------------

interface IBRDF {
    float3 evaluate_diffuse(float3 N, float3 L, float3 V,
                            float3 base_color, float metallic);
    float3 evaluate_specular(float3 N, float3 L, float3 V,
                             float3 base_color, float metallic, float roughness);
};

// ---------------------------------------------------------------------------
// GGX helper functions
// ---------------------------------------------------------------------------

float D_GGX(float NdotH, float a2) {
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float G_SmithSchlick(float NdotV, float NdotL, float a2) {
    float k = sqrt(a2) * 0.5;
    float g_v = NdotV / (NdotV * (1.0 - k) + k);
    float g_l = NdotL / (NdotL * (1.0 - k) + k);
    return g_v * g_l;
}

float3 F_Schlick(float VdotH, float3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}

// ---------------------------------------------------------------------------
// Cook-Torrance GGX BRDF
// ---------------------------------------------------------------------------

struct CookTorranceGGX : IBRDF {
    float3 evaluate_diffuse(float3 N, float3 L, float3 V,
                            float3 base_color, float metallic) {
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) return float3(0.0);

        float3 H = normalize(L + V);
        float VdotH = max(dot(V, H), 0.0);
        float3 F0 = lerp(float3(0.04), base_color, metallic);
        float3 F = F_Schlick(VdotH, F0);

        return (1.0 - F) * (1.0 - metallic) * base_color / PI * NdotL;
    }

    float3 evaluate_specular(float3 N, float3 L, float3 V,
                             float3 base_color, float metallic, float roughness) {
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) return float3(0.0);

        float3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float NdotV = max(dot(N, V), 0.001);
        float VdotH = max(dot(V, H), 0.0);

        float alpha = max(roughness * roughness, 0.001);
        float a2 = alpha * alpha;

        float3 F0 = lerp(float3(0.04), base_color, metallic);

        float  D = D_GGX(NdotH, a2);
        float  G = G_SmithSchlick(NdotV, NdotL, a2);
        float3 F = F_Schlick(VdotH, F0);

        return D * G * F / (4.0 * NdotV * NdotL + 0.001) * NdotL;
    }
};

// ---------------------------------------------------------------------------
// Light sampling — representative point technique (Karis 2013)
// ---------------------------------------------------------------------------

struct LightSample {
    float3 diffuse_L;           // center direction for diffuse
    float3 spec_L;              // representative direction for specular
    float attenuation;
    float spec_roughness;       // roughness widened by light angular extent
    bool valid;
};

LightSample sample_distant_light(GpuLight light, float3 N, float3 V, float roughness) {
    LightSample ls;
    ls.valid = true;
    ls.attenuation = 1.0;
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
            ortho = normalize(cross(center_L, float3(0, 1, 0)));
        }
        ls.spec_L = center_L * cos_half + ortho * sin(half_angle);
    }

    // Convolve GGX NDF with the light's angular distribution:
    //   alpha' = sqrt(alpha^2 + sin^2(half_angle))
    float alpha = roughness * roughness;
    float sin_half = sin(half_angle);
    float alpha_prime = sqrt(alpha * alpha + sin_half * sin_half);
    ls.spec_roughness = sqrt(clamp(alpha_prime, 0.001, 1.0));

    return ls;
}

LightSample sample_area_light(GpuLight light, float3 P, float3 N, float3 V, float roughness) {
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

    // Representative point: closest point on sphere to reflection ray
    float3 R = reflect(-V, N);
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
    ls.attenuation = 1.0 / max(dist * dist, 0.001);

    // Convolve GGX NDF with the light's apparent angular size
    float alpha = roughness * roughness;
    float sin_half = effective_radius / center_dist;
    float alpha_prime = sqrt(alpha * alpha + sin_half * sin_half);
    ls.spec_roughness = sqrt(clamp(alpha_prime, 0.001, 1.0));

    return ls;
}

LightSample sample_light(GpuLight light, float3 P, float3 N, float3 V, float roughness) {
    if (light.type == LIGHT_DOME) {
        LightSample ls;
        ls.valid = false;
        ls.diffuse_L = float3(0.0);
        ls.spec_L = float3(0.0);
        ls.attenuation = 0.0;
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
    StructuredBuffer<GpuLight> lights, uint light_count,
    float3 P, float3 N, float3 V,
    float3 base_color, float metallic, float roughness
) {
    float3 result = float3(0.0);
    for (uint i = 0; i < light_count; i++) {
        GpuLight light = lights[i];

        if (light.type == LIGHT_DOME) {
            result += light.color * light.intensity * base_color * 0.2;
            continue;
        }

        LightSample ls = sample_light(light, P, N, V, roughness);
        if (!ls.valid) continue;

        float3 diffuse  = brdf.evaluate_diffuse(N, ls.diffuse_L, V, base_color, metallic);
        float3 specular = brdf.evaluate_specular(N, ls.spec_L, V, base_color, metallic, ls.spec_roughness);
        result += light.color * light.intensity * ls.attenuation * (diffuse + specular);
    }
    return result;
}
