// GPU-independent shadow helpers shared with C++ through shader_pure.
// Reconstruct linear light-space depth from zero-to-one NDC depth.
public float linearize_shadow_depth(float ndc_z, float near_plane, float far_plane,
                                    uint projection_type) {
    if (projection_type == 0u) {
        return near_plane + ndc_z * (far_plane - near_plane);
    }
    float denom = far_plane - ndc_z * (far_plane - near_plane);
    return (near_plane * far_plane) / (denom < 1e-6f ? 1e-6f : denom);
}
