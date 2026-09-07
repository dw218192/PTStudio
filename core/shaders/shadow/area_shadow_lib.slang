// Omnidirectional PCSS from one representative point at the emitter center.
// Cube faces remove the spotlight-frustum cutoff. All sampling is in world
// space so the emitter kernel and receiver plane remain continuous at seams.
import shadow_sampling_lib;

internal uint cube_face(float3 d) {
    float3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) return d.x >= 0.0 ? 0u : 1u;
    if (a.y >= a.z) return d.y >= 0.0 ? 2u : 3u;
    return d.z >= 0.0 ? 4u : 5u;
}

internal float3 cube_ray(float2 uv, uint face) {
    float2 p = uv * 2.0 - 1.0;
    if (face == 0u) return float3(1, p.y, -p.x);
    if (face == 1u) return float3(-1, p.y, p.x);
    if (face == 2u) return float3(p.x, 1, -p.y);
    if (face == 3u) return float3(p.x, -1, p.y);
    if (face == 4u) return float3(p.x, p.y, 1);
    return float3(-p.x, p.y, -1);
}

internal float2 cube_uv(float3 d, uint face) {
    float2 p;
    if (face == 0u) p = float2(-d.z, d.y) / abs(d.x);
    else if (face == 1u) p = float2(d.z, d.y) / abs(d.x);
    else if (face == 2u) p = float2(d.x, -d.z) / abs(d.y);
    else if (face == 3u) p = float2(d.x, d.z) / abs(d.y);
    else if (face == 4u) p = float2(d.x, d.y) / abs(d.z);
    else p = float2(-d.x, d.y) / abs(d.z);
    return p * 0.5 + 0.5;
}

// Returns a blocker projected onto the center ray, or -1 for an unoccluded
// tap. Intersect the receiver plane at the actual sampled TEXEL CENTER;
// correcting only the requested UV leaves repeating texel-sized acne bands.
internal float area_blocker(Texture2DArray<float> map, ShadowInfo si,
                            float3 receiver, float3 normal, float3 direction) {
    if (dot(direction, direction) < 1e-12) return -1.0;
    uint face = cube_face(direction);
    int resolution = int(round(1.0 / si.texel_size));
    int2 texel = clamp(int2(cube_uv(direction, face) * float(resolution)),
                        int2(0), int2(resolution - 1));
    float3 ray = cube_ray((float2(texel) + 0.5) * si.texel_size, face);
    float plane = dot(normal, receiver);
    float denom = dot(normal, ray);
    // The receiver-plane approximation is invalid past its horizon. Treat
    // these taps as lit rather than comparing extrapolated depths at infinity.
    if (plane * denom <= 0.0 || abs(denom) < 1e-6) return -1.0;
    float receiver_depth = plane / denom;
    float z = map.Load(int4(texel, int(si.layer + face), 0)).x;
    if (z >= 1.0 || receiver_depth <= si.light_near) return -1.0;
    float blocker_depth = linearize_shadow_depth(z, si.light_near, si.light_far, 1u);
    // Small precision bias, independent of filter radius. Plane correction
    // already handles the large slope contribution at wide PCSS radii.
    float bias = max(0.0002, receiver_depth * si.texel_size * 0.25);
    if (blocker_depth >= receiver_depth - bias) return -1.0;
    float distance = dot(ray * blocker_depth, normalize(receiver));
    return distance > 0.0 ? distance : -1.0;
}

internal float2 area_sample(uint i, uint count, float2 shift, bool disk) {
    // Stratified low-discrepancy samples with a per-pixel temporal shift.
    // Keep rect samples aligned to the emitter (rotating a square changes it).
    float2 xi = frac(float2((float(i) + 0.5) / float(count),
                              float(i) * 0.61803398875) + shift);
    if (!disk) return xi * 2.0 - 1.0;
    float angle = xi.y * 6.2831853;
    return sqrt(xi.x) * float2(cos(angle), sin(angle));
}

public float compute_area_shadow(Texture2DArray<float> map, ShadowInfo si,
                                  float3 world_pos, float3 normal,
                                  float2 pixel, uint frame) {
    float3 receiver = world_pos - si.light_position.xyz;
    float distance = length(receiver);
    if (distance <= si.light_near) return 1.0;
    // Integer pixel hash plus an irrational temporal progression avoids
    // rotating a sparse fixed disk into coherent rings and bands.
    uint h = uint(pixel.x) * 0x27d4eb2du ^ uint(pixel.y) * 0x165667b1u;
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13;
    float2 shift = frac(float2(float(h & 65535u) / 65536.0,
                                 float(h >> 16) / 65536.0) +
                          float(frame % 32u) * float2(0.754877666, 0.569840296));
    bool disk = si.light_position.w > 0.5;
    float sum = 0.0;
    float count = 0.0;
    float center = area_blocker(map, si, receiver, normal, receiver);
    // Bound the broad search to keep cost fixed for near-emitter receivers.
    // The second stage estimates the physical penumbra from actual blockers.
    float search_scale = center > 0.0 ? clamp(distance / center - 1.0, 1.0, 4.0) : 1.5;
    for (uint i = 0; i < 32u; ++i) {
        float2 xi = area_sample(i, 32u, shift, disk);
        float3 offset = si.light_u.xyz * xi.x + si.light_v.xyz * xi.y;
        float blocker = area_blocker(map, si, receiver, normal, receiver + offset * search_scale);
        if (blocker > 0.0) { sum += blocker; count += 1.0; }
    }
    if (count == 0.0 && center < 0.0) return 1.0;
    float blocker = count > 0.0 ? sum / count : center;
    float penumbra = clamp(distance / max(blocker, si.light_near) - 1.0, 0.0, 8.0);
    float visible = 0.0;
    float total_weight = 0.0;
    float3 light_cross = cross(si.light_u.xyz, si.light_v.xyz);
    float cross_length_sq = dot(light_cross, light_cross);
    float3 emission = -light_cross * rsqrt(max(cross_length_sq, 1e-12));
    // One-texel minimum footprint for contact edges / zero-softness lights.
    float3 axis = normalize(receiver);
    float3 tangent = normalize(cross(axis, abs(axis.y) < 0.9 ? float3(0, 1, 0) : float3(1, 0, 0)));
    float3 bitangent = cross(axis, tangent);
    for (uint i = 0; i < 64u; ++i) {
        float2 xi = area_sample(i, 64u, frac(shift + 0.37), disk);
        float3 emitter = si.light_u.xyz * xi.x + si.light_v.xyz * xi.y;
        float3 to_light = emitter - receiver;
        float light_distance_sq = dot(to_light, to_light);
        float3 L = to_light * rsqrt(max(light_distance_sq, 1e-8));
        // Diffuse geometric contribution: do not count emitter samples below
        // the receiver horizon as missing shadow coverage. LTC still evaluates
        // the actual BRDF; this is only the scalar visibility approximation.
        float weight = max(dot(normal, L), 0.0) * max(dot(emission, -L), 0.0) / max(light_distance_sq, 1e-8);
        if (cross_length_sq < 1e-12) weight = 1.0;
        float3 offset = emitter * penumbra;
        offset += (tangent * xi.x + bitangent * xi.y) * distance * si.texel_size;
        visible += area_blocker(map, si, receiver, normal, receiver + offset) < 0.0 ? weight : 0.0;
        total_weight += weight;
    }
    return total_weight > 1e-8 ? visible / total_weight : 1.0;
}
