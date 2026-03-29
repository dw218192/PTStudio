import bvh;
import lighting;
import brdf;
import light;
import material_sampling;
import sampling;

// ── Scene data ──────────────────────────────────────────────────

struct PackedTriangle {
    float3 v0; uint _p0;
    float3 v1; uint _p1;
    float3 v2; uint _p2;
    float3 n0; uint _p3;
    float3 n1; uint _p4;
    float3 n2; uint _pad6;
    float2 uv0; float2 uv1;
    float2 uv2; uint _p5[2];
};

struct PTUniforms {
    float3 camera_pos;
    uint frame_index;
    float4x4 inv_vp;
    uint width;
    uint height;
    uint instance_count;
    uint light_count;
    uint total_frames;
    uint tlas_node_count;
    uint _pad0;
    uint _pad1;
    float3 dome_modulation;
};

[[vk::binding(0, 0)]] ConstantBuffer<PTUniforms> u;
[[vk::binding(1, 0)]] StructuredBuffer<PackedTriangle> triangles;
[[vk::binding(2, 0)]] StructuredBuffer<Material> materials;
[[vk::binding(3, 0)]] StructuredBuffer<Light> lights;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> accumulation;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> output_buf;

[[vk::binding(6, 0)]] StructuredBuffer<BVHNode> bvh_nodes;

[[vk::binding(7, 0)]] Texture2DArray<float4> scene_textures;
[[vk::binding(8, 0)]] SamplerState scene_sampler;

[[vk::binding(9, 0)]] StructuredBuffer<Instance> instances;

[[vk::binding(0, 1)]] TextureCube<float4> env_cubemap;
[[vk::binding(1, 1)]] SamplerState env_sampler;

static const float PT_EPS = 0.0001;

// ── Ray-triangle intersection ───────────────────────────────────

struct HitRecord {
    float3 position;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float2 uv;
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

    // TLAS traversal (world space)
    uint tlas_stack[32];
    int tlas_sp = 0;
    tlas_stack[tlas_sp++] = 0;  // TLAS root

    while (tlas_sp > 0) {
        BVHNode tlas_node = bvh_nodes[tlas_stack[--tlas_sp]];
        if (!ray_aabb(ro, rd, tlas_node.aabb_min, tlas_node.aabb_max, t_best))
            continue;

        if (tlas_node.count > 0) {
            // TLAS leaf — iterate instances
            for (uint i = 0; i < tlas_node.count; i++) {
                Instance inst = instances[tlas_node.left_first + i];

                // Transform ray to local space (don't normalize — preserves t)
                float3 local_ro = mul(inst.inv_transform, float4(ro, 1.0)).xyz;
                float3 local_rd = mul(inst.inv_transform, float4(rd, 0.0)).xyz;

                // BLAS traversal (local space)
                uint blas_stack[32];
                int blas_sp = 0;
                blas_stack[blas_sp++] = inst.blas_offset;

                while (blas_sp > 0) {
                    BVHNode blas_node = bvh_nodes[blas_stack[--blas_sp]];
                    if (!ray_aabb(local_ro, local_rd,
                                  blas_node.aabb_min, blas_node.aabb_max, t_best))
                        continue;

                    if (blas_node.count > 0) {
                        // BLAS leaf — test triangles
                        for (uint j = 0; j < blas_node.count; j++) {
                            PackedTriangle tri =
                                triangles[inst.tri_offset + blas_node.left_first + j];
                            float bu, bv;
                            float t = intersect_triangle(local_ro, local_rd, tri, bu, bv);
                            if (t > 0.0 && t < t_best) {
                                t_best = t;
                                best.hit = true;
                                float3 local_hit = local_ro + local_rd * t;
                                best.position =
                                    mul(inst.transform, float4(local_hit, 1.0)).xyz;
                                float3 local_normal = normalize(
                                    tri.n0 * (1.0 - bu - bv) + tri.n1 * bu + tri.n2 * bv);
                                float3x3 normal_mat =
                                    transpose(float3x3(inst.inv_transform));
                                best.normal =
                                    normalize(mul(normal_mat, local_normal));
                                best.uv = tri.uv0 * (1.0 - bu - bv)
                                        + tri.uv1 * bu + tri.uv2 * bv;
                                best.material_index = inst.material_index;

                                // Tangent frame from triangle UVs
                                float3 dp1 = tri.v1 - tri.v0;
                                float3 dp2 = tri.v2 - tri.v0;
                                float2 duv1 = tri.uv1 - tri.uv0;
                                float2 duv2 = tri.uv2 - tri.uv0;
                                float uv_det = duv1.x * duv2.y - duv1.y * duv2.x;
                                if (abs(uv_det) > 1e-8) {
                                    float inv_det = 1.0 / uv_det;
                                    float3 local_T = (dp1 * duv2.y - dp2 * duv1.y) * inv_det;
                                    float3 local_B = (dp2 * duv1.x - dp1 * duv2.x) * inv_det;
                                    float3x3 M = float3x3(inst.transform);
                                    best.tangent = normalize(mul(M, local_T));
                                    best.bitangent = normalize(mul(M, local_B));
                                } else {
                                    TangentFrame tf = TangentFrame::build(best.normal);
                                    best.tangent = tf.T;
                                    best.bitangent = tf.B;
                                }
                            }
                        }
                    } else {
                        // BLAS interior — push children
                        blas_stack[blas_sp++] = blas_node.left_first;
                        blas_stack[blas_sp++] = blas_node.left_first + 1;
                    }
                }
            }
        } else {
            // TLAS interior — push children
            tlas_stack[tlas_sp++] = tlas_node.left_first;
            tlas_stack[tlas_sp++] = tlas_node.left_first + 1;
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
            // Miss → sample environment cubemap
            float3 env_color = env_cubemap.SampleLevel(env_sampler, ray_dir, 0.0).rgb;
            radiance += throughput * env_color * u.dome_modulation;
            break;
        }

        float3 N = hit.normal;
        float3 V = -ray_dir;
        if (dot(N, V) < 0.0) N = -N;

        // Material
        float3 base_color = float3(0.8);
        float metallic = 0.0;
        float roughness = 0.5;
        if (has_texture(hit.material_index)) {
            Material mat = materials[hit.material_index];
            MaterialSample sm = sample_material_lod(mat, hit.uv, scene_textures, scene_sampler, 0.0);
            base_color = sm.base_color;
            metallic = sm.metallic;
            roughness = sm.roughness;
            N = apply_normal_map(N, hit.tangent, hit.bitangent, mat, hit.uv,
                                 scene_textures, scene_sampler, 0.0);
        }

        // Direct lighting (NEE with shadow rays)
        float3 direct = float3(0.0);
        for (uint li = 0; li < u.light_count; li++) {
            Light light = lights[li];
            if (light.type == LIGHT_DOME) continue;

            LightSampleStochastic ls = sample_direct(light, hit.position, rng);
            if (!ls.valid || ls.pdf < 1e-4) continue;

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
        if (pdf < 1e-4 || wi_local.z <= 0.0) break;

        ShadingGeometry g = ShadingGeometry::from_local(wi_local, wo_local, roughness);
        float3 f = brdf.eval(g, base_color, metallic);
        float3 bounce_weight = f * wi_local.z / pdf;
        // Clamp per-bounce weight to suppress fireflies from degenerate GGX lobes
        float max_comp = max(bounce_weight.x, max(bounce_weight.y, bounce_weight.z));
        if (max_comp > 10.0) bounce_weight *= 10.0 / max_comp;
        throughput *= bounce_weight;

        ray_origin = hit.position + N * PT_EPS;
        ray_dir = frame.to_world(wi_local);
    }


    // Accumulate
    float4 prev = (u.total_frames == 1) ? float4(0.0) : accumulation[pixel_index];
    float4 acc = prev + float4(radiance, 1.0);
    accumulation[pixel_index] = acc;
    output_buf[pixel_index] = float4(acc.xyz / float(u.total_frames), 1.0);
}
