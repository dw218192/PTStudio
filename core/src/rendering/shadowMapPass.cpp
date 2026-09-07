#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/webgpu/device.h>
#include <imgui.h>
#include <shadow_map_shader_metadata.h>

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>

namespace pts::rendering {

namespace {

ShadowProjection distant_projection(const LightData& light, const glm::vec3& aabb_min,
                                    const glm::vec3& aabb_max, float softness) {
    auto dir = glm::normalize(light.direction);
    auto center = (aabb_min + aabb_max) * 0.5f;
    auto half_diag = glm::length(aabb_max - aabb_min) * 0.5f;

    auto up = glm::vec3(0, 1, 0);
    if (std::abs(glm::dot(dir, up)) > 0.99f) up = glm::vec3(1, 0, 0);
    auto light_view = glm::lookAt(center - dir * half_diag, center, up);

    glm::vec3 ls_min(std::numeric_limits<float>::max());
    glm::vec3 ls_max(std::numeric_limits<float>::lowest());
    for (int c = 0; c < 8; ++c) {
        glm::vec3 corner((c & 1) ? aabb_max.x : aabb_min.x, (c & 2) ? aabb_max.y : aabb_min.y,
                         (c & 4) ? aabb_max.z : aabb_min.z);
        auto ls_pt = glm::vec3(light_view * glm::vec4(corner, 1.0f));
        ls_min = glm::min(ls_min, ls_pt);
        ls_max = glm::max(ls_max, ls_pt);
    }

    ShadowProjection out;
    auto& info = out.info;
    info.light_near = -ls_max.z;
    info.light_far = -ls_min.z;
    auto proj = glm::ortho(ls_min.x, ls_max.x, ls_min.y, ls_max.y, info.light_near, info.light_far);
    info.light_vp = proj * light_view;
    float ortho_width = ls_max.x - ls_min.x;
    if (ortho_width > 0.0f) {
        float half_angle = glm::radians(std::max(light.angle, 0.0f) * 0.5f);
        info.light_size_uv = std::tan(half_angle) / ortho_width * softness;
    }
    info.has_shadow = 1;
    out.layer_vps[0] = info.light_vp;
    out.layer_count = 1;
    return out;
}

ShadowProjection area_projection(const LightData& light, const glm::vec3& aabb_min,
                                 const glm::vec3& aabb_max, float softness, glm::vec2 half_size,
                                 bool disk) {
    // One representative point, with six faces to retain grazing-angle coverage.
    static const glm::vec3 directions[] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                           {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    static const glm::vec3 ups[] = {{0, -1, 0}, {0, -1, 0}, {0, 0, 1},
                                    {0, 0, -1}, {0, -1, 0}, {0, -1, 0}};
    glm::vec3 position(light.transform[3]);
    ShadowProjection out;
    auto& info = out.info;
    info.light_far = 0.01f;
    for (int c = 0; c < 8; ++c) {
        glm::vec3 corner((c & 1) ? aabb_max.x : aabb_min.x, (c & 2) ? aabb_max.y : aabb_min.y,
                         (c & 4) ? aabb_max.z : aabb_min.z);
        info.light_far = std::max(info.light_far, glm::length(corner - position));
    }
    info.light_near = std::max(0.001f, info.light_far * 0.0001f);
    auto proj = glm::perspective(glm::radians(90.0f), 1.0f, info.light_near, info.light_far);
    out.layer_count = 6;
    for (uint32_t face = 0; face < out.layer_count; ++face) {
        auto view = glm::lookAt(position, position + directions[face], ups[face]);
        out.layer_vps[face] = proj * view;
    }
    info.light_vp = out.layer_vps[0];
    info.projection_type = 2;
    info.has_shadow = 1;
    info.light_position = glm::vec4(position, static_cast<float>(disk));
    info.light_u =
        glm::vec4(glm::vec3(light.transform[0]) * std::max(half_size.x, 0.0f) * softness, 0);
    info.light_v =
        glm::vec4(glm::vec3(light.transform[1]) * std::max(half_size.y, 0.0f) * softness, 0);
    return out;
}

}  // namespace

ShadowProjection compute_shadow_projection(const LightData& light, const glm::vec3& aabb_min,
                                           const glm::vec3& aabb_max, bool pcss) {
    if (!light.casts_shadow) return {};
    float softness = pcss ? std::max(light.shadow_pcss_softness, 0.0f) : 0.0f;
    switch (light.type) {
        case LightData::Type::Distant:
            return distant_projection(light, aabb_min, aabb_max, softness);
        case LightData::Type::Rect:
            return area_projection(light, aabb_min, aabb_max, softness,
                                   {light.width * 0.5f, light.height * 0.5f}, false);
        case LightData::Type::Disk:
            return area_projection(light, aabb_min, aabb_max, softness, glm::vec2(light.radius),
                                   true);
        case LightData::Type::Sphere:
        case LightData::Type::Dome:
            return {};
    }
    UNREACHABLE();
}

ShadowMapPass::Outputs ShadowMapPass::add_to_frame_graph(FrameGraph& fg, const PassContext& ctx,
                                                         const Inputs&) {
    PTS_ZONE_SCOPED;
    ensure_initialized(ctx.device);

    auto desc_layout = fg.bind_group_layout(
        "shadow_map/desc", shadow_map_shader::create_bind_group_layout_0(ctx.device.handle()));

    // Position-only vertex layout: stride=12, one Float32x3 at offset 0, location 0
    WGPUVertexAttribute pos_attr{};
    pos_attr.format = WGPUVertexFormat_Float32x3;
    pos_attr.offset = 0;
    pos_attr.shaderLocation = 0;

    auto* pipeline_handle = fg.render_pipeline("shadow_map")
                                .shader("core/generated/shaders/shadow/shadow_map.wgsl")
                                .no_fragment()
                                .depth_format(WGPUTextureFormat_Depth32Float)
                                .depth_write(true)
                                .depth_compare(WGPUCompareFunction_Less)
                                .cull_mode(WGPUCullMode_Front)
                                // Rasterizer-level slope-scale bias keeps grazing-angle
                                // receivers from self-shadowing. Constant=1 pushes every
                                // shadowed fragment one ulp away; slope=2 scales with
                                // depth gradient (|dz/dx|, |dz/dy|).
                                .depth_bias(1, 2.0f)
                                .vertex_buffer({12, WGPUVertexStepMode_Vertex, {pos_attr}})
                                .bind_group_layouts({desc_layout})
                                .build();

    auto lights = ctx.world.get_lights().span_raw();
    auto info_count = std::max(uint32_t(1), static_cast<uint32_t>(lights.size()));
    std::vector<ShadowInfo> infos(info_count);
    std::vector<glm::mat4> layer_vps;
    auto scene_bounds = ctx.world.scene_bounds();
    uint32_t shadow_count = 0;
    uint32_t map_count = 0;
    bool has_distant = false;
    if (m_enabled) {
        for (uint32_t li = 0; li < static_cast<uint32_t>(lights.size()); ++li) {
            if (!lights[li].active) continue;
            auto projection = compute_shadow_projection(lights[li].value, scene_bounds.min,
                                                        scene_bounds.max, m_pcss);
            if (projection.layer_count == 0) continue;
            infos[li] = projection.info;
            infos[li].layer = map_count;
            layer_vps.insert(layer_vps.end(), projection.layer_vps.begin(),
                             projection.layer_vps.begin() + projection.layer_count);
            map_count += projection.layer_count;
            has_distant |= projection.info.projection_type == 0;
            if (++shadow_count >= k_max_shadow_maps) break;
        }
    }

    // Always ensure at least 1 layer for downstream descriptors
    uint32_t layer_count = std::max(map_count, 1u);
    // A 1024 cube has comparable angular density to the old 2048 / 120-deg
    // map, with complete coverage in 24 MiB. Preserve distant-light density
    // when both kinds share the array.
    uint32_t resolution = has_distant ? m_resolution : std::max(m_resolution / 2, 1u);

    // Register shadow texture array with frame graph
    TextureDesc shadow_tex_desc;
    shadow_tex_desc.width = resolution;
    shadow_tex_desc.height = resolution;
    shadow_tex_desc.array_layers = layer_count;
    shadow_tex_desc.format = WGPUTextureFormat_Depth32Float;
    shadow_tex_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                                          WGPUTextureUsage_TextureBinding);
    shadow_tex_desc.force_array_view = true;
    auto shadow_array = create_texture(fg, shadow_tex_desc, "shadow_depth_array");

    // Register shadow info buffer (one ShadowInfo per light, minimum 1)
    uint64_t info_bytes = static_cast<uint64_t>(info_count) * sizeof(ShadowInfo);
    BufferDesc info_buf_desc;
    info_buf_desc.size = info_bytes;
    info_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    auto shadow_info_buf = create_buffer(fg, info_buf_desc, "shadow_info");

    if (shadow_count == 0) {
        // Upload all-inactive ShadowInfo entries
        auto queue = ctx.queue;
        fg.add_pass("shadow_info_upload")
            .execute([queue, shadow_info_buf, infos = std::move(infos)](ExecuteContext& exec,
                                                                        WGPUComputePassEncoder) {
                auto buf = exec.get(shadow_info_buf).buffer;
                wgpuQueueWriteBuffer(queue, buf, 0, infos.data(),
                                     infos.size() * sizeof(ShadowInfo));
            });
        return {shadow_array, shadow_info_buf};
    }

    for (auto& info : infos) {
        if (!info.has_shadow) continue;
        info.texel_size = 1.0f / static_cast<float>(resolution);
        // World-space normal offset limits PCF bleeding at sharp creases.
        info.normal_bias = 0.02f;
    }
    auto objects = ctx.world.get_objects().span_raw();
    uint32_t total_slots = static_cast<uint32_t>(objects.size());

    // Model buffer: one model matrix per object (shared across all layers)
    uint64_t model_buf_size =
        std::max(uint64_t(1), static_cast<uint64_t>(total_slots)) * k_uniform_align;
    BufferDesc model_buf_desc;
    model_buf_desc.size = model_buf_size;
    model_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto model_buf_decl = create_buffer(fg, model_buf_desc, "models");

    // Light VP buffer: one VP matrix per shadow layer
    uint64_t vp_buf_size =
        std::max(uint64_t(1), static_cast<uint64_t>(map_count)) * k_uniform_align;
    BufferDesc vp_buf_desc;
    vp_buf_desc.size = vp_buf_size;
    vp_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto vp_buf_decl = create_buffer(fg, vp_buf_desc, "light_vps");

    // Descriptor: binding 0 = model (dynamic), binding 1 = light VP (dynamic)
    auto desc_decl = descriptor(fg, desc_layout, "desc0")
                         .buffer(0, model_buf_decl, 0, 64)
                         .buffer(1, vp_buf_decl, 0, 64)
                         .build();

    auto queue = ctx.queue;
    const auto& world = ctx.world;

    // Upload shadow info + model matrices + light VPs in a single compute pass
    fg.add_pass("shadow_upload")
        .execute([queue, shadow_info_buf, model_buf_decl, vp_buf_decl, map_count,
                  infos = std::move(infos), layer_vps = std::move(layer_vps),
                  &world](ExecuteContext& exec, WGPUComputePassEncoder) {
            // Shadow info buffer
            auto info_buf = exec.get(shadow_info_buf).buffer;
            wgpuQueueWriteBuffer(queue, info_buf, 0, infos.data(),
                                 infos.size() * sizeof(ShadowInfo));

            // Model matrices (uploaded once, shared across all layers)
            auto model_buf = exec.get(model_buf_decl).buffer;
            auto objs = world.get_objects().span_raw();
            for (uint32_t oi = 0; oi < static_cast<uint32_t>(objs.size()); ++oi) {
                if (!objs[oi].active) continue;
                if (!objs[oi].value.visible) continue;
                wgpuQueueWriteBuffer(queue, model_buf, oi * k_uniform_align,
                                     &objs[oi].value.transform, sizeof(glm::mat4));
            }

            // Light VP matrices
            auto vp_buf = exec.get(vp_buf_decl).buffer;
            for (uint32_t l = 0; l < map_count; ++l) {
                wgpuQueueWriteBuffer(queue, vp_buf, l * k_uniform_align, &layer_vps[l],
                                     sizeof(glm::mat4));
            }
        });

    // Render each shadow layer
    for (uint32_t layer = 0; layer < map_count; ++layer) {
        fg.add_pass("shadow_depth_" + std::to_string(layer))
            .depth(shadow_array, layer)
            .execute([=, &world](ExecuteContext& exec, WGPURenderPassEncoder pass) {
                auto desc = exec.get(desc_decl).bind_group;
                auto objs = world.get_objects().span_raw();
                auto mesh_slots = world.get_meshes().span_raw();
                uint32_t slots = static_cast<uint32_t>(objs.size());

                uint32_t vp_offset = layer * k_uniform_align;
                wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
                for (uint32_t i = 0; i < slots; ++i) {
                    if (!objs[i].active) continue;
                    if (!objs[i].value.visible) continue;
                    uint32_t model_offset = i * k_uniform_align;
                    uint32_t dyn_offsets[2] = {model_offset, vp_offset};
                    wgpuRenderPassEncoderSetBindGroup(pass, 0, desc, 2, dyn_offsets);
                    const auto& mesh = mesh_slots[objs[i].value.mesh_index].value;
                    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh.position_buffer.handle(), 0,
                                                         mesh.position_buffer.size());
                    wgpuRenderPassEncoderSetIndexBuffer(pass, mesh.index_buffer.handle(),
                                                        WGPUIndexFormat_Uint32, 0,
                                                        mesh.index_buffer.size());
                    wgpuRenderPassEncoderDrawIndexed(pass, mesh.index_count, 1, 0, 0, 0);
                }
            });
    }

    return {shadow_array, shadow_info_buf};
}

void ShadowMapPass::draw_imgui() {
    ImGui::Checkbox("Enabled", &m_enabled);
    ImGui::Checkbox("PCSS", &m_pcss);
}

}  // namespace pts::rendering
