import bvh;
import lighting;
import brdf;
import light;
import sampling;

// ── Scene data ──────────────────────────────────────────────────

struct PackedTriangle {
    float3 v0; uint _p0;
    float3 v1; uint _p1;
    float3 v2; uint _p2;
    float3 n0; uint _p3;
    float3 n1; uint _p4;
    float3 n2; uint material_index;
    uint _p5[4];
};

struct PTUniforms {
    float3 camera_pos;
    uint frame_index;
    float4x4 inv_vp;
    uint width;
    uint height;
    uint triangle_count;
    uint light_count;
    uint total_frames;
};

[[vk::binding(0, 0)]] ConstantBuffer<PTUniforms> u;
[[vk::binding(1, 0)]] StructuredBuffer<PackedTriangle> triangles;
[[vk::binding(2, 0)]] StructuredBuffer<Material> materials;
[[vk::binding(3, 0)]] StructuredBuffer<Light> lights;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> accumulation;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> output_buf;

[[vk::binding(6, 0)]] StructuredBuffer<BVHNode> bvh_nodes;

static const float PT_EPS = 0.0001;

// ── Ray-triangle intersection ───────────────────────────────────

struct HitRecord {
    float3 position;
    float3 normal;
    uint material_index;
    bool hit;
};

float intersect_triangle(float3 ro, float3 rd, PackedTriangle tri,
                         out float bu, out float bv) {
    bu = 0.0; bv = 0.0;
    float3 e1 = tri.v1 - tri.v0, e2 = tri.v2 - tri.v0;
    float3 h = cross(rd, e2);
    float det = dot(e1, h);
    if (abs(det) < PT_EPS) return -1.0;
    float inv = 1.0 / det;
    float3 s = ro - tri.v0;
    bu = inv * dot(s, h);
    if (bu < 0.0 || bu > 1.0) return -1.0;
    float3 q = cross(s, e1);
    bv = inv * dot(rd, q);
    if (bv < 0.0 || bu + bv > 1.0) return -1.0;
    float t = inv * dot(e2, q);
    return t > PT_EPS ? t : -1.0;
}

HitRecord trace_scene(float3 ro, float3 rd) {
    HitRecord best;
    best.hit = false;
    float t_best = 1e30;

    uint stack[32];
    int sp = 0;
    stack[sp++] = 0;  // root

    while (sp > 0) {
        BVHNode node = bvh_nodes[stack[--sp]];
        if (!ray_aabb(ro, rd, node.aabb_min, node.aabb_max, t_best)) continue;
        if (node.count > 0) {
            // Leaf — test triangles
            for (uint i = 0; i < node.count; i++) {
                PackedTriangle tri = triangles[node.left_first + i];
                float bu, bv;
                float t = intersect_triangle(ro, rd, tri, bu, bv);
                if (t > 0.0 && t < t_best) {
                    t_best = t;
                    best.hit = true;
                    best.position = ro + rd * t;
                    best.normal = normalize(
                        tri.n0 * (1.0 - bu - bv) + tri.n1 * bu + tri.n2 * bv);
                    best.material_index = tri.material_index;
                }
            }
        } else {
            // Interior — push both children
            stack[sp++] = node.left_first;
            stack[sp++] = node.left_first + 1;
        }
    }
    return best;
}

// ── Main ────────────────────────────────────────────────────────

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 did : SV_DispatchThreadID) {
    if (did.x >= u.width || did.y >= u.height) return;

    static const uint MAX_BOUNCES = 5;
    uint pixel_index = did.y * u.width + did.x;

    Rng rng = Rng::create(pixel_index * 1973u + u.frame_index * 2069168927u);

    // Camera ray
    float2 jitter = float2(rng.next(), rng.next());
    float2 uv = (float2(did.xy) + jitter) / float2(u.width, u.height);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 nc = mul(u.inv_vp, float4(ndc, 0.0, 1.0));
    float4 fc = mul(u.inv_vp, float4(ndc, 1.0, 1.0));
    float3 ray_origin = u.camera_pos;
    float3 ray_dir = normalize(fc.xyz / fc.w - nc.xyz / nc.w);

    float3 throughput = float3(1.0);
    float3 radiance = float3(0.0);
    CookTorranceGGX brdf;

    for (uint bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        HitRecord hit = trace_scene(ray_origin, ray_dir);
        if (!hit.hit) {
            // Miss → dome light contributes as environment
            for (uint li = 0; li < u.light_count; li++) {
                if (lights[li].type == LIGHT_DOME) {
                    radiance += throughput * lights[li].color * lights[li].intensity;
                }
            }
            break;
        }

        float3 N = hit.normal;
        float3 V = -ray_dir;
        if (dot(N, V) < 0.0) N = -N;

        // Material
        float3 base_color = float3(0.8);
        float metallic = 0.0;
        float roughness = 0.5;
        if (hit.material_index != 0xFFFFFFFF) {
            Material mat = materials[hit.material_index];
            base_color = mat.diffuse_color;
            metallic = mat.metallic;
            roughness = mat.roughness;
        }

        // Direct lighting (NEE with shadow rays)
        float3 direct = float3(0.0);
        for (uint li = 0; li < u.light_count; li++) {
            Light light = lights[li];
            if (light.type == LIGHT_DOME) continue;

            LightSampleStochastic ls = sample_direct(light, hit.position, rng);
            if (!ls.valid || ls.pdf <= 0.0) continue;

            float3 wi = normalize(ls.position - hit.position);
            float NdotL = dot(N, wi);
            if (NdotL <= 0.0) continue;

            // Shadow ray
            float dist_to_light = length(ls.position - hit.position);
            HitRecord shadow_hit = trace_scene(hit.position + N * PT_EPS, wi);
            if (shadow_hit.hit) {
                float shadow_dist = length(shadow_hit.position - hit.position);
                if (shadow_dist < dist_to_light - PT_EPS) continue;
            }

            ShadingGeometry g = ShadingGeometry::from_world(N, wi, V, roughness);
            float3 f = brdf.eval(g, base_color, metallic);
            direct += f * ls.radiance * NdotL / ls.pdf;
        }
        radiance += throughput * direct;

        // Russian roulette (after first bounce)
        if (bounce > 0) {
            float survival = min(max(throughput.x, max(throughput.y, throughput.z)), 0.95);
            if (rng.next() > survival) break;
            throughput /= survival;
        }

        // Sample BRDF for next bounce
        TangentFrame frame = TangentFrame::build(N);
        float3 wo_local = frame.to_local(V);
        float3 wi_local = sample_brdf(rng, wo_local, roughness);
        float pdf = pdf_brdf(wi_local, wo_local, roughness);
        if (pdf <= 0.0 || wi_local.z <= 0.0) break;

        ShadingGeometry g = ShadingGeometry::from_local(wi_local, wo_local, roughness);
        float3 f = brdf.eval(g, base_color, metallic);
        throughput *= f * wi_local.z / pdf;

        ray_origin = hit.position + N * PT_EPS;
        ray_dir = frame.to_world(wi_local);
    }

    // Accumulate
    float4 prev = (u.total_frames == 1) ? float4(0.0) : accumulation[pixel_index];
    float4 acc = prev + float4(radiance, 1.0);
    accumulation[pixel_index] = acc;
    output_buf[pixel_index] = float4(acc.xyz / float(u.total_frames), 1.0);
}
