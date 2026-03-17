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
    float angle;
};

static const uint LIGHT_DISTANT = 0;
static const uint LIGHT_SPHERE  = 1;
static const uint LIGHT_RECT    = 2;
static const uint LIGHT_DISK    = 3;
static const uint LIGHT_DOME    = 4;

struct LightSample {
    float3 L;
    float attenuation;
    float spec_norm;
    bool valid;
};

// Returns BRDF * (N.L) -- the combined outgoing radiance contribution for this light direction.
// Returns float3(0) when N.L <= 0 (light below horizon).
float3 eval_brdf(float3 N, float3 L, float3 V, float3 base_color, float roughness) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return float3(0.0);

    float3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);

    float alpha = lerp(128.0, 16.0, roughness);
    float k_s = 1.0 - roughness;

    float3 diffuse = base_color / 3.14159265;

    float spec_norm = (alpha + 4.0) / (8.0 * 3.14159265);
    float3 specular = float3(k_s * spec_norm * pow(NdotH, alpha));

    return (diffuse + specular) * NdotL;
}

LightSample sample_light(GpuLight light, float3 P, float3 N, float3 V, float roughness) {
    LightSample ls;
    ls.valid = true;
    ls.spec_norm = 1.0;

    if (light.type == LIGHT_DOME) {
        ls.valid = false;
        ls.L = float3(0.0);
        ls.attenuation = 0.0;
        return ls;
    }

    if (light.type == LIGHT_DISTANT) {
        ls.L = -light.direction_or_pos;
        ls.attenuation = 1.0;
        return ls;
    }

    float3 center = light.direction_or_pos;
    float effective_radius;

    if (light.type == LIGHT_SPHERE) {
        effective_radius = light.radius;
    } else if (light.type == LIGHT_RECT) {
        effective_radius = sqrt(light.width * light.height) / 2.0;
    } else {
        // LIGHT_DISK
        effective_radius = light.radius;
    }

    // Representative point technique for sphere-like lights
    float3 R = reflect(-V, N);
    float3 to_center = center - P;
    float center_to_ray = dot(to_center, R);
    float3 closest_on_ray = P + R * max(center_to_ray, 0.0);
    float3 closest_to_center = center - closest_on_ray;
    float closest_dist = length(closest_to_center);

    float3 representative_point;
    if (closest_dist > effective_radius) {
        representative_point = center - normalize(closest_to_center) * effective_radius;
    } else {
        representative_point = closest_on_ray;
    }

    float3 to_rep = representative_point - P;
    float dist = length(to_rep);
    ls.L = to_rep / max(dist, 0.001);
    ls.attenuation = 1.0 / max(dist * dist, 0.001);

    float alpha = lerp(128.0, 16.0, roughness);
    float center_dist = max(length(to_center), 0.001);
    ls.spec_norm = pow(saturate(alpha / (alpha + effective_radius / (2.0 * center_dist))), 2.0);

    return ls;
}

float3 evaluate_lighting(
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

        float3 brdf_ndotl = eval_brdf(N, ls.L, V, base_color, roughness);
        result += light.color * light.intensity * ls.attenuation * ls.spec_norm * brdf_ndotl;
    }
    return result;
}
