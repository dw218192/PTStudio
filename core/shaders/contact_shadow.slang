import light;
import light_iteration;

struct ContactShadowUniforms {
    float4x4 projection;
    float4x4 inv_projection;
    float4x4 view;
    float2 viewport_size;
    float max_distance;
    float thickness;
    float normal_offset;
    int step_count;
    uint light_count;
    uint _pad;
};

// GBuffer consumer slots (0-3)
[[vk::binding(0, 0)]] [NonFilterable] Texture2D<float>  depth_tex;
[[vk::binding(1, 0)]] [NonFiltering]  SamplerState      depth_sampler;
[[vk::binding(2, 0)]] Texture2D<float4> normals_tex;
[[vk::binding(3, 0)]] SamplerState      linear_sampler;
// ContactShadow-specific (4-5)
[[vk::binding(4, 0)]] ConstantBuffer<ContactShadowUniforms> u;
[[vk::binding(5, 0)]] StructuredBuffer<Light> lights;

struct VsOut {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD;
};

[shader("vertex")]
VsOut vs_main(uint vertex_id : SV_VertexID) {
    VsOut output;
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = float2(uv.x, 1.0 - uv.y);
    return output;
}

float3 reconstruct_view_pos(float2 uv, float depth) {
    float2 ndc_xy = float2(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);
    float4 clip = float4(ndc_xy, depth, 1.0);
    float4 view = mul(u.inv_projection, clip);
    return view.xyz / view.w;
}

float2 project_to_uv(float3 view_pos) {
    float4 clip = mul(u.projection, float4(view_pos, 1.0));
    if (clip.w <= 0.0) return float2(-1.0);
    clip.xyz /= clip.w;
    return float2(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
}

float3 get_light_direction_view(Light light, float3 frag_view_pos) {
    if (light.type == LIGHT_DISTANT) {
        // Transform light direction to view space (w=0 for direction)
        float3 world_dir = -light.direction_or_pos;
        return normalize(mul(u.view, float4(world_dir, 0.0)).xyz);
    }
    // Point/area lights: world position → view space (w=1), then subtract fragment
    float3 light_view_pos = mul(u.view, float4(light.direction_or_pos, 1.0)).xyz;
    return normalize(light_view_pos - frag_view_pos);
}

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD) : SV_Target0 {
    float depth = depth_tex.SampleLevel(depth_sampler, uv, 0).x;
    if (depth >= 1.0) {
        return float4(1.0, 0.0, 0.0, 0.0);
    }

    float3 frag_pos = reconstruct_view_pos(uv, depth);

    // Decode view-space normal from RG16Float
    float2 n_xy = normals_tex.SampleLevel(linear_sampler, uv, 0).xy;
    float n_z = sqrt(max(0.0, 1.0 - n_xy.x * n_xy.x - n_xy.y * n_xy.y));
    float3 normal = normalize(float3(n_xy, n_z));

    // Offset origin along normal to avoid self-shadowing
    float3 origin = frag_pos + normal * u.normal_offset;

    float combined_shadow = 1.0;

    uint visible_count = get_light_count(u.light_count, uv);
    for (uint iter = 0; iter < visible_count; iter++) {
        uint li = get_light_index(iter, uv);
        Light light = lights[li];

        if (light.type == LIGHT_DOME) continue;

        float3 to_light = get_light_direction_view(light, frag_pos);

        // Skip if light is below the surface
        if (dot(to_light, normal) <= 0.0) continue;

        float step_size = u.max_distance / float(u.step_count);
        bool occluded = false;

        for (int s = 1; s <= u.step_count; s++) {
            float3 march_pos = origin + to_light * (step_size * float(s));

            float2 sample_uv = project_to_uv(march_pos);
            if (any(sample_uv < 0.0) || any(sample_uv > 1.0)) break;

            float sample_depth = depth_tex.SampleLevel(depth_sampler, sample_uv, 0).x;
            float3 sample_view_pos = reconstruct_view_pos(sample_uv, sample_depth);

            float depth_diff = sample_view_pos.z - march_pos.z;
            if (depth_diff > 0.0 && depth_diff < u.thickness) {
                occluded = true;
                break;
            }
        }

        if (occluded) {
            combined_shadow = 0.0;
            break;
        }
    }

    return float4(combined_shadow, 0.0, 0.0, 0.0);
}
