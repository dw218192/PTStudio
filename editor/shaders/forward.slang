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
    float angle;  // distant light angular extent in degrees
};

struct Uniforms {
    float4x4 mvp;
    float4x4 model;
    float3 camera_pos;
    float time;
    uint material_index;
    uint light_count;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> u;

[[vk::binding(1, 0)]]
StructuredBuffer<Material> materials;

[[vk::binding(2, 0)]]
StructuredBuffer<GpuLight> lights;

struct VsIn {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float3 color    : COLOR;
    float2 uv       : TEXCOORD;
};

struct VsOut {
    float4 position     : SV_Position;
    float3 world_normal : NORMAL;
    float3 world_pos    : WORLD_POS;
    float3 color        : COLOR;
    float2 uv           : TEXCOORD;
};

[shader("vertex")]
VsOut vs_main(VsIn input) {
    VsOut output;
    output.position = mul(u.mvp, float4(input.position, 1.0));
    output.world_normal = normalize(mul(u.model, float4(input.normal, 0.0)).xyz);
    output.world_pos = mul(u.model, float4(input.position, 1.0)).xyz;
    output.color = input.color;
    output.uv = input.uv;
    return output;
}

static const uint LIGHT_DISTANT = 0;
static const uint LIGHT_SPHERE  = 1;
static const uint LIGHT_RECT    = 2;
static const uint LIGHT_DISK    = 3;
static const uint LIGHT_DOME    = 4;

[shader("fragment")]
float4 fs_main(float3 world_normal : NORMAL, float3 world_pos : WORLD_POS,
               float3 color : COLOR) : SV_Target0 {
    float3 N = normalize(world_normal);
    float3 V = normalize(u.camera_pos - world_pos);

    float3 base_color;
    float roughness = 0.5;
    if (u.material_index == 0xFFFFFFFF) {
        base_color = color;
    } else {
        base_color = materials[u.material_index].diffuse_color;
        roughness = materials[u.material_index].roughness;
    }

    float shininess = lerp(128.0, 16.0, roughness);
    float3 result = float3(0.0);

    for (uint i = 0; i < u.light_count; i++) {
        GpuLight light = lights[i];

        float3 L;
        float attenuation = 1.0;

        if (light.type == LIGHT_DOME) {
            result += light.color * light.intensity * base_color * 0.2;
            continue;
        } else if (light.type == LIGHT_DISTANT) {
            L = -light.direction_or_pos;
        } else {
            float3 to_light = light.direction_or_pos - world_pos;
            float dist2 = dot(to_light, to_light);
            L = normalize(to_light);
            attenuation = 1.0 / max(dist2, 0.001);
        }

        float3 H = normalize(L + V);
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);

        // Distant light angular extent: soften the terminator.
        // angle is in degrees; convert half-angle to cosine threshold.
        if (light.type == LIGHT_DISTANT && light.angle > 0.0) {
            float half_angle_rad = radians(light.angle * 0.5);
            float cos_thresh = cos(half_angle_rad);
            NdotL = smoothstep(cos_thresh, 1.0, NdotL + (1.0 - cos_thresh));
        }

        float spec = pow(NdotH, shininess) * (1.0 - roughness);

        result += light.color * light.intensity * attenuation
                  * (base_color * NdotL + spec);
    }

    result += 0.03 * base_color;

    return float4(result, 1.0);
}
