/// GPU BVH node — matches C++ BVHNode (32 bytes).
public struct BVHNode {
    public float3 aabb_min;
    public uint left_first;
    public float3 aabb_max;
    public uint count;          // 0 = interior, >0 = leaf tri count
};

/// Per-instance data for two-level BVH traversal — matches C++ GPUInstance (144 bytes).
public struct Instance {
    public float4x4 transform;
    public float4x4 inv_transform;
    public uint blas_offset;
    public uint tri_offset;
    public uint tri_count;
    public uint material_index;
};

/// Ray-AABB slab test. Returns true if the ray enters the box before t_best.
public bool ray_aabb(float3 ro, float3 rd, float3 bmin, float3 bmax, float t_best) {
    float3 inv_rd = 1.0 / rd;
    float3 t0 = (bmin - ro) * inv_rd;
    float3 t1 = (bmax - ro) * inv_rd;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float enter = max(max(tmin.x, tmin.y), tmin.z);
    float exit  = min(min(tmax.x, tmax.y), tmax.z);
    return enter <= exit && exit >= 0.0 && enter < t_best;
}
