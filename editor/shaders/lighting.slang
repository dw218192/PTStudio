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
    // Returns BRDF * (N.L) for the given light direction.
    // Must return float3(0) when N.L <= 0.
    float3 evaluate(float3 N, float3 L, float3 V, float3 base_color, float roughness);
};

// ---------------------------------------------------------------------------
// Normalized Blinn-Phong BRDF
//   f_r = rho/pi + k_s * (alpha+4)/(8*pi) * (N.H)^alpha
// ---------------------------------------------------------------------------

struct BlinnPhong : IBRDF {
    float3 evaluate(float3 N, float3 L, float3 V, float3 base_color, float roughness) {
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) return float3(0.0);

        float3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);

        float alpha = lerp(128.0, 16.0, roughness);
        float k_s = 1.0 - roughness;

        float3 diffuse = base_color / PI;
        float spec_norm = (alpha + 4.0) / (8.0 * PI);
        float3 specular = float3(k_s * spec_norm * pow(NdotH, alpha));

        return (diffuse + specular) * NdotL;
    }
};

// ---------------------------------------------------------------------------
// Light sampling — representative point technique
// ---------------------------------------------------------------------------

struct LightSample {
    float3 L;
    float attenuation;
    float spec_norm;
    bool valid;
};

LightSample sample_distant_light(GpuLight light, float3 N, float3 V, float roughness) {
    LightSample ls;
    ls.valid = true;
    ls.attenuation = 1.0;
    ls.spec_norm = 1.0;

    float3 center_L = -light.direction_or_pos;
    float half_angle = radians(light.angle * 0.5);

    if (half_angle < 0.001) {
        ls.L = center_L;
        return ls;
    }

    // Representative direction: closest point on the light cone to R
    float3 R = reflect(-V, N);
    float cos_delta = dot(R, center_L);
    float cos_half = cos(half_angle);

    if (cos_delta >= cos_half) {
        ls.L = R;
    } else {
        float3 ortho = R - center_L * cos_delta;
        float ortho_len = length(ortho);
        if (ortho_len > 0.0001) {
            ortho /= ortho_len;
        } else {
            ortho = normalize(cross(center_L, float3(0, 1, 0)));
        }
        ls.L = center_L * cos_half + ortho * sin(half_angle);
    }

    // Energy normalization
    float alpha = lerp(128.0, 16.0, roughness);
    float sin_half = sin(half_angle);
    ls.spec_norm = pow(saturate(alpha / (alpha + sin_half * 2.0)), 2.0);

    return ls;
}

LightSample sample_area_light(GpuLight light, float3 P, float3 N, float3 V, float roughness) {
    LightSample ls;
    ls.valid = true;

    float3 center = light.direction_or_pos;
    float effective_radius;

    if (light.type == LIGHT_SPHERE) {
        effective_radius = light.radius;
    } else if (light.type == LIGHT_RECT) {
        effective_radius = sqrt(light.width * light.height) / 2.0;
    } else {
        effective_radius = light.radius;
    }

    // Representative point: closest point on sphere to reflection ray
    float3 R = reflect(-V, N);
    float3 to_center = center - P;
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
    ls.L = to_rep / max(dist, 0.001);
    ls.attenuation = 1.0 / max(dist * dist, 0.001);

    // Specular energy normalization (Karis 2013)
    float alpha = lerp(128.0, 16.0, roughness);
    float center_dist = max(length(to_center), 0.001);
    ls.spec_norm = pow(saturate(alpha / (alpha + effective_radius / (2.0 * center_dist))), 2.0);

    return ls;
}

LightSample sample_light(GpuLight light, float3 P, float3 N, float3 V, float roughness) {
    if (light.type == LIGHT_DOME) {
        LightSample ls;
        ls.valid = false;
        ls.L = float3(0.0);
        ls.attenuation = 0.0;
        ls.spec_norm = 1.0;
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
    float3 P, float3 N, float3 V, float3 base_color, float roughness
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

        float3 brdf_ndotl = brdf.evaluate(N, ls.L, V, base_color, roughness);
        result += light.color * light.intensity * ls.attenuation * ls.spec_norm * brdf_ndotl;
    }
    return result;
}
