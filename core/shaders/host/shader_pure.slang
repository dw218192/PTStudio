// GPU-independent, CPU-testable API exported to C++.
// Keep algorithms in the same modules used by GPU shaders.
import shadow_sampling_lib;
import shadow_pure;
import brdf_lib;
import sampling_lib;

export __extern_cpp float pts_linearize_shadow_depth(float ndc, float near_plane,
                                                     float far_plane, uint projection) {
    return linearize_shadow_depth(ndc, near_plane, far_plane, projection);
}

export __extern_cpp float pts_pcss_penumbra_uv(float receiver, float blocker,
                                              float size, uint projection) {
    return pcss_penumbra_uv(receiver, blocker, size, projection);
}

export __extern_cpp float pts_fresnel_dielectric(float cosine, float eta) {
    return fresnel_dielectric(cosine, eta);
}

export __extern_cpp float pts_temporal_ema_blend(float current, float history, float alpha) {
    return temporal_ema_blend(current, history, alpha);
}

export __extern_cpp float pts_clamp_history(float history, float mean, float stddev, float gamma) {
    return clamp_history_to_neighborhood(history, mean, stddev, gamma);
}

export __extern_cpp void pts_tangent_roundtrip(float x, float y, float z,
                                              out float rx, out float ry, out float rz) {
    TangentFrame frame = TangentFrame.build(normalize(float3(1, 2, 3)));
    float3 result = frame.to_world(frame.to_local(float3(x, y, z)));
    rx = result.x;
    ry = result.y;
    rz = result.z;
}

export __extern_cpp void pts_sample_brdf(uint seed, float roughness,
                                        out float x, out float y, out float z, out float pdf) {
    Rng rng = Rng.create(seed);
    float3 wo = float3(0, 0, 1);
    float3 wi = sample_brdf(rng, wo, roughness);
    x = wi.x;
    y = wi.y;
    z = wi.z;
    pdf = pdf_brdf(wi, wo, roughness);
}
