// Authoritative shadow buffer declarations. C++ upload types are generated
// from the WGSL bindings that consume these types.
public struct ShadowInfo {
    public float4x4 light_vp;
    public float texel_size;
    public float normal_bias;
    public uint has_shadow;
    public uint layer;
    public float light_near;
    public float light_far;
    public float light_size_uv;
    public uint projection_type; // 0 = ortho, 1 = perspective, 2 = cube
    public float4 light_position; // xyz = cube origin, w = disk (1) or rect (0)
    public float4 light_u; // emitter half-axis, softness included
    public float4 light_v;
};

public struct ShadowVisibilityUniforms {
    public float4x4 inv_view_proj;
    public float2 viewport_size;
    public uint shadow_light_index;
    public uint frame_index;
};
