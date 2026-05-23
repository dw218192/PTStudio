import brdf_lib;
import sampling_lib;

public struct Material {
    public float3 diffuse_color;
    public float metallic;
    public float3 emissive_color;
    public float roughness;
    public float opacity;
    public uint diffuse_tex;
    public uint normal_tex;
    public uint metallic_tex;
    public uint roughness_tex;
    public uint emissive_tex;
    public uint opacity_tex;
    public float ior;
    public float opacity_threshold;
    public uint tex_channels;
    public uint light_index;  // GPU light index for proxy meshes (0xFFFFFFFF = none)
    public uint _pad0;
};

/// Returns true if P is on the emitting (front) side of a directional area light.
/// Area lights emit along -Z local, i.e. -cross(right, up) in world space.
public bool is_front_facing(float3 P, float3 center, float3 right_vec, float3 up_vec) {
    float3 light_normal = -normalize(cross(right_vec, up_vec));
    return dot(P - center, light_normal) > 0.0;
}

public static const uint LIGHT_DISTANT = 0;
public static const uint LIGHT_SPHERE  = 1;
public static const uint LIGHT_RECT    = 2;
public static const uint LIGHT_DISK    = 3;
public static const uint LIGHT_DOME    = 4;

// ---------------------------------------------------------------------------
// Light sampling -- representative point technique (Karis 2013)
// ---------------------------------------------------------------------------

public struct LightSample {
    public float3 diffuse_L;           // center direction for diffuse
    public float3 spec_L;              // representative direction for specular
    public float diffuse_attenuation;  // 1/d^2 from light center
    public float spec_attenuation;     // 1/d^2 from representative point
    public float spec_roughness;       // roughness widened by light angular extent
    public bool valid;
};

public struct Light {
    public float3 direction_or_pos;
    public uint type;
    public float3 color;
    public float intensity;
    public float3 right;
    public float radius;
    public float3 up;
    public float angle;  // distant light angular extent in degrees
};

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Front-face emission normal: -cross(right, up), normalized
public float3 light_emission_normal(Light light) {
    return -normalize(cross(light.right, light.up));
}

/// Surface area of the light
/// Rect: 4 * length(right) * length(up)   (right/up are half-extents)
/// Disk: PI * radius * radius
/// Sphere: 4 * PI * radius * radius
/// Distant/Dome: 0 (not applicable)
public float light_area(Light light) {
    static const float PI = 3.14159265358979323846;
    if (light.type == LIGHT_RECT) {
        return 4.0 * length(light.right) * length(light.up);
    } else if (light.type == LIGHT_DISK) {
        return PI * light.radius * light.radius;
    } else if (light.type == LIGHT_SPHERE) {
        return 4.0 * PI * light.radius * light.radius;
    }
    return 0.0;
}

// ---------------------------------------------------------------------------
// Representative point sampling (Karis 2013) -- free functions
// ---------------------------------------------------------------------------

internal LightSample sample_representative_distant(Light light, float3 N, float3 V, float roughness) {
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
            float3 u = abs(center_L.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
            ortho = normalize(cross(center_L, u));
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

internal LightSample sample_representative_area(Light light, float3 P, float3 N, float3 V, float roughness) {
    LightSample ls;
    ls.valid = true;
    ls.spec_roughness = roughness;

    float3 center = light.direction_or_pos;
    float effective_radius;

    if (light.type == LIGHT_SPHERE) {
        effective_radius = light.radius;
    } else if (light.type == LIGHT_RECT) {
        effective_radius = sqrt(length(light.right) * length(light.up));
    } else {
        effective_radius = light.radius;
    }

    float3 to_center = center - P;
    float center_dist = max(length(to_center), 0.001);

    // Directional area lights (rect, disk) only emit from their front face.
    if ((light.type == LIGHT_RECT || light.type == LIGHT_DISK) && !is_front_facing(P, center, light.right, light.up)) {
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

/// Sample a light using the representative point technique (Karis 2013).
/// Dispatches by light type.
public LightSample sample_representative(Light light, float3 P, float3 N, float3 V, float roughness) {
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
        return sample_representative_distant(light, N, V, roughness);
    }

    return sample_representative_area(light, P, N, V, roughness);
}

// ---------------------------------------------------------------------------
// Stochastic direct light sampling (for path tracer NEE)
// ---------------------------------------------------------------------------

internal static const float k_distant_light_dist = 1e6;

public struct LightSampleStochastic {
    public float3 position;    // sampled point on light surface
    public float3 normal;      // light surface normal at sampled point
    public float3 radiance;    // emitted radiance (color * intensity)
    public float pdf;          // probability density in SOLID ANGLE measure
    public bool valid;
};

/// Sample a uniform direction on the unit sphere.
internal float3 uniform_sphere_direction(inout Rng rng) {
    float u1 = rng.next();
    float u2 = rng.next();
    float z = 1.0 - 2.0 * u1;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * u2;
    return float3(r * cos(phi), r * sin(phi), z);
}

/// Convert area PDF to solid-angle PDF.
internal float pdf_area_to_solid_angle(float pdf_area, float dist, float cos_theta_light) {
    if (cos_theta_light < 1e-8) return 0.0;
    return pdf_area * dist * dist / cos_theta_light;
}

// --- Per-type sample_direct implementations ---

internal LightSampleStochastic sample_direct_rect(Light light, float3 P, inout Rng rng) {
    LightSampleStochastic result;
    float3 normal = light_emission_normal(light);

    // Front-face check: shading point must be on emitting side
    if (dot(P - light.direction_or_pos, normal) <= 0.0) {
        result.valid = false;
        return result;
    }

    float u1 = rng.next();
    float u2 = rng.next();
    float3 sampled_pos = light.direction_or_pos
        + light.right * (2.0 * u1 - 1.0)
        + light.up * (2.0 * u2 - 1.0);

    float3 to_light = sampled_pos - P;
    float dist = length(to_light);
    float3 wi = to_light / dist;

    float pdf_area = 1.0 / light_area(light);
    float cos_theta_light = abs(dot(normal, -wi));

    result.position = sampled_pos;
    result.normal = normal;
    result.radiance = light.color * light.intensity;
    result.pdf = pdf_area_to_solid_angle(pdf_area, dist, cos_theta_light);
    result.valid = result.pdf > 0.0;
    return result;
}

internal LightSampleStochastic sample_direct_disk(Light light, float3 P, inout Rng rng) {
    LightSampleStochastic result;
    float3 normal = light_emission_normal(light);

    // Front-face check
    if (dot(P - light.direction_or_pos, normal) <= 0.0) {
        result.valid = false;
        return result;
    }

    float u1 = rng.next();
    float u2 = rng.next();
    float r = light.radius * sqrt(u1);
    float theta = 2.0 * PI * u2;
    float3 sampled_pos = light.direction_or_pos
        + normalize(light.right) * (r * cos(theta))
        + normalize(light.up) * (r * sin(theta));

    float3 to_light = sampled_pos - P;
    float dist = length(to_light);
    float3 wi = to_light / dist;

    float pdf_area = 1.0 / light_area(light);
    float cos_theta_light = abs(dot(normal, -wi));

    result.position = sampled_pos;
    result.normal = normal;
    result.radiance = light.color * light.intensity;
    result.pdf = pdf_area_to_solid_angle(pdf_area, dist, cos_theta_light);
    result.valid = result.pdf > 0.0;
    return result;
}

internal LightSampleStochastic sample_direct_sphere(Light light, float3 P, inout Rng rng) {
    LightSampleStochastic result;
    float3 center = light.direction_or_pos;
    float center_dist = length(center - P);

    // P inside sphere: invalid
    if (center_dist < light.radius) {
        result.valid = false;
        return result;
    }

    // Uniform point on full sphere, reject backfacing
    float3 dir = uniform_sphere_direction(rng);
    float3 sampled_pos = center + light.radius * dir;
    float3 normal = dir;

    float3 to_light = sampled_pos - P;
    float dist = length(to_light);
    float3 wi = to_light / dist;

    float cos_theta_light = dot(normal, -wi);
    if (cos_theta_light <= 0.0) {
        result.valid = false;
        return result;
    }

    float pdf_area = 1.0 / light_area(light);

    result.position = sampled_pos;
    result.normal = normal;
    result.radiance = light.color * light.intensity;
    result.pdf = pdf_area_to_solid_angle(pdf_area, dist, cos_theta_light);
    result.valid = result.pdf > 0.0;
    return result;
}

internal LightSampleStochastic sample_direct_distant(Light light, float3 P, inout Rng rng) {
    LightSampleStochastic result;
    float half_angle = radians(light.angle * 0.5);

    if (half_angle < 1e-5) {
        // Delta / point light approximation
        float3 wi = -light.direction_or_pos;
        result.position = P + wi * k_distant_light_dist;
        result.normal = -wi;
        result.radiance = light.color * light.intensity;
        result.pdf = 1.0;
        result.valid = true;
        return result;
    }

    // Uniform cone sampling
    float cos_half = cos(half_angle);
    float u1 = rng.next();
    float u2 = rng.next();
    float cos_theta = 1.0 - u1 * (1.0 - cos_half);
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float phi = 2.0 * PI * u2;

    // Local direction in cone frame (z = light direction)
    float3 local_dir = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    // Transform to world space: z axis = -light.direction_or_pos
    float3 z_axis = -normalize(light.direction_or_pos);
    TangentFrame frame = TangentFrame::build(z_axis);
    float3 wi = frame.to_world(local_dir);

    result.position = P + wi * k_distant_light_dist;
    result.normal = -wi;
    result.radiance = light.color * light.intensity;
    result.pdf = 1.0 / (2.0 * PI * (1.0 - cos_half));
    result.valid = true;
    return result;
}

/// Sample a random point on the light surface visible from P.
/// Returns position, normal, radiance, and solid-angle PDF.
public LightSampleStochastic sample_direct(Light light, float3 P, inout Rng rng) {
    if (light.type == LIGHT_RECT)    return sample_direct_rect(light, P, rng);
    if (light.type == LIGHT_DISK)    return sample_direct_disk(light, P, rng);
    if (light.type == LIGHT_SPHERE)  return sample_direct_sphere(light, P, rng);
    if (light.type == LIGHT_DISTANT) return sample_direct_distant(light, P, rng);

    // LIGHT_DOME: not supported for direct sampling
    LightSampleStochastic result;
    result.valid = false;
    return result;
}

/// Evaluate the solid-angle PDF for a given direction wi from point P.
public float pdf_direct(Light light, float3 P, float3 wi) {
    if (light.type == LIGHT_DOME) return 0.0;

    if (light.type == LIGHT_DISTANT) {
        float half_angle = radians(light.angle * 0.5);
        if (half_angle < 1e-5) return 1.0;
        float cos_half = cos(half_angle);
        float3 light_dir = -normalize(light.direction_or_pos);
        float cos_theta = dot(wi, light_dir);
        if (cos_theta < cos_half) return 0.0;
        return 1.0 / (2.0 * PI * (1.0 - cos_half));
    }

    if (light.type == LIGHT_RECT) {
        float3 normal = light_emission_normal(light);
        float denom = dot(normal, wi);
        if (abs(denom) < 1e-8) return 0.0;
        float t = dot(light.direction_or_pos - P, normal) / denom;
        if (t <= 0.0) return 0.0;

        float3 hit = P + wi * t;
        float3 offset = hit - light.direction_or_pos;
        float u = dot(offset, normalize(light.right)) / length(light.right);
        float v = dot(offset, normalize(light.up)) / length(light.up);
        if (abs(u) > 1.0 || abs(v) > 1.0) return 0.0;

        float pdf_area = 1.0 / light_area(light);
        float cos_theta_light = dot(normal, -wi);
        if (cos_theta_light <= 0.0) return 0.0;
        return pdf_area_to_solid_angle(pdf_area, t, cos_theta_light);
    }

    if (light.type == LIGHT_DISK) {
        float3 normal = light_emission_normal(light);
        float denom = dot(normal, wi);
        if (abs(denom) < 1e-8) return 0.0;
        float t = dot(light.direction_or_pos - P, normal) / denom;
        if (t <= 0.0) return 0.0;

        float3 hit = P + wi * t;
        if (length(hit - light.direction_or_pos) > light.radius) return 0.0;

        float pdf_area = 1.0 / light_area(light);
        float cos_theta_light = dot(normal, -wi);
        if (cos_theta_light <= 0.0) return 0.0;
        return pdf_area_to_solid_angle(pdf_area, t, cos_theta_light);
    }

    if (light.type == LIGHT_SPHERE) {
        float3 center = light.direction_or_pos;
        float3 oc = P - center;
        float b = dot(oc, wi);
        float c = dot(oc, oc) - light.radius * light.radius;
        float disc = b * b - c;
        if (disc < 0.0) return 0.0;
        float t = -b - sqrt(disc);
        if (t <= 0.0) return 0.0;

        float3 hit = P + wi * t;
        float3 normal = normalize(hit - center);
        float cos_theta_light = dot(normal, -wi);
        if (cos_theta_light <= 0.0) return 0.0;

        float pdf_area = 1.0 / light_area(light);
        return pdf_area_to_solid_angle(pdf_area, t, cos_theta_light);
    }

    return 0.0;
}
