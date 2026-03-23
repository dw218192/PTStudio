import brdf;

struct Material {
    float3 diffuse_color;
    float metallic;
    float roughness;
    float opacity;
    float2 _pad;
};

/// Returns true if P is on the emitting (front) side of a directional area light.
/// Area lights emit along -Z local, i.e. -cross(right, up) in world space.
bool is_front_facing(float3 P, float3 center, float3 right_vec, float3 up_vec) {
    float3 light_normal = -normalize(cross(right_vec, up_vec));
    return dot(P - center, light_normal) > 0.0;
}

static const uint LIGHT_DISTANT = 0;
static const uint LIGHT_SPHERE  = 1;
static const uint LIGHT_RECT    = 2;
static const uint LIGHT_DISK    = 3;
static const uint LIGHT_DOME    = 4;

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

struct Light {
    float3 direction_or_pos;
    uint type;
    float3 color;
    float intensity;
    float3 right;
    float radius;
    float3 up;
    float angle;  // distant light angular extent in degrees

    LightSample sample_distant(float3 N, float3 V, float roughness) {
        LightSample ls;
        ls.valid = true;
        ls.diffuse_attenuation = 1.0;
        ls.spec_attenuation = 1.0;
        ls.spec_roughness = roughness;

        float3 center_L = -direction_or_pos;
        ls.diffuse_L = center_L;

        float half_angle = radians(angle * 0.5);

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

    LightSample sample_area(float3 P, float3 N, float3 V, float roughness) {
        LightSample ls;
        ls.valid = true;
        ls.spec_roughness = roughness;

        float3 center = direction_or_pos;
        float effective_radius;

        if (type == LIGHT_SPHERE) {
            effective_radius = radius;
        } else if (type == LIGHT_RECT) {
            effective_radius = sqrt(length(right) * length(up));
        } else {
            effective_radius = radius;
        }

        float3 to_center = center - P;
        float center_dist = max(length(to_center), 0.001);

        // Directional area lights (rect, disk) only emit from their front face.
        if ((type == LIGHT_RECT || type == LIGHT_DISK) && !is_front_facing(P, center, right, up)) {
            ls.valid = false;
            return ls;
        }

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

    LightSample sample(float3 P, float3 N, float3 V, float roughness) {
        if (type == LIGHT_DOME) {
            LightSample ls;
            ls.valid = false;
            ls.diffuse_L = float3(0.0);
            ls.spec_L = float3(0.0);
            ls.diffuse_attenuation = 0.0;
            ls.spec_attenuation = 0.0;
            ls.spec_roughness = roughness;
            return ls;
        }

        if (type == LIGHT_DISTANT) {
            return sample_distant(N, V, roughness);
        }

        return sample_area(P, N, V, roughness);
    }
};
