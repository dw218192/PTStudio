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
    float light_offset;
    int step_count;
    uint light_count;
};

// GBuffer consumer slots (0-3)
[[vk::binding(0, 0)]] [NonFiltering] Texture2D<float>  depth_tex;
[[vk::binding(1, 0)]] [NonFiltering]  SamplerState     depth_sampler;
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

float3 get_light_direction_view(Light light, float3 frag_view_pos) {
    if (light.type == LIGHT_DISTANT) {
        // Transform light direction to view space (w=0 for direction)
        float3 world_dir = -light.direction_or_pos;
        return normalize(mul(u.view, float4(world_dir, 0.0)).xyz);
    }
    // Point/area lights: world position -> view space (w=1), then subtract fragment
    float3 light_view_pos = mul(u.view, float4(light.direction_or_pos, 1.0)).xyz;
    return normalize(light_view_pos - frag_view_pos);
}

// Hash a 2D coordinate to a jitter in [0, 1). Used to break up march banding.
float hash12(float2 p) {
    return frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
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

    float combined_shadow = 1.0;
    float jitter = hash12(uv * u.viewport_size);

    uint visible_count = get_light_count(u.light_count, uv);
    for (uint iter = 0; iter < visible_count; iter++) {
        uint li = get_light_index(iter, uv);
        Light light = lights[li];

        if (light.type == LIGHT_DOME) continue;

        float3 to_light = get_light_direction_view(light, frag_pos);

        // Skip if light is below the surface
        float ndotl = dot(to_light, normal);
        if (ndotl <= 0.0) continue;

        // Grazing-angle bias: scale normal offset by (1 - NdotL); additional
        // light_offset nudges the march start along the light ray to prevent
        // the first samples from intersecting the originating surface.
        float normal_bias = u.normal_offset * (1.0 + 2.0 * (1.0 - ndotl));
        float3 p0_view = frag_pos + normal * normal_bias + to_light * u.light_offset;
        float3 p1_view = p0_view + to_light * u.max_distance;

        // Project ray endpoints to UV. If either endpoint is behind the
        // camera we skip; a more robust impl would clip the ray.
        float4 c0 = mul(u.projection, float4(p0_view, 1.0));
        float4 c1 = mul(u.projection, float4(p1_view, 1.0));
        if (c0.w <= 0.0 || c1.w <= 0.0) continue;

        float2 uv0 = float2(c0.x / c0.w * 0.5 + 0.5, 0.5 - c0.y / c0.w * 0.5);
        float2 uv1 = float2(c1.x / c1.w * 0.5 + 0.5, 0.5 - c1.y / c1.w * 0.5);

        // Under perspective projection, 1/z_view is linear across screen
        // space, so we interpolate it along the UV-linear march.
        float inv_z0 = 1.0 / p0_view.z;
        float inv_z1 = 1.0 / p1_view.z;

        float shadow = 1.0;
        for (int s = 1; s <= u.step_count; s++) {
            float t = (float(s) - jitter) / float(u.step_count);
            float2 sample_uv = lerp(uv0, uv1, t);
            if (any(sample_uv < 0.0) || any(sample_uv > 1.0)) break;

            float sample_depth = depth_tex.SampleLevel(depth_sampler, sample_uv, 0).x;
            if (sample_depth >= 1.0) continue;

            float ray_view_z = 1.0 / lerp(inv_z0, inv_z1, t);
            float sample_view_z = reconstruct_view_pos(sample_uv, sample_depth).z;
            float depth_diff = sample_view_z - ray_view_z;

            if (depth_diff > 0.0 && depth_diff < u.thickness) {
                // Soft falloff: hits near ray end shadow less (step_fade),
                // hits near the thickness edge shadow less (thickness_fade).
                float step_fade = 1.0 - t;
                float thickness_fade = 1.0 - saturate(depth_diff / u.thickness);
                shadow = 1.0 - step_fade * thickness_fade;
                break;
            }
        }

        combined_shadow = min(combined_shadow, shadow);
    }

    return float4(combined_shadow, 0.0, 0.0, 0.0);
}
