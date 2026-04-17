#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/shadowLightProjection.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/webgpu/device.h>
#include <imgui.h>
#include <shadow_shader_metadata.h>

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace pts::rendering {

namespace {

bool casts_shadow_map(LightData::Type type) {
    return type == LightData::Type::Distant || type == LightData::Type::Rect ||
           type == LightData::Type::Disk;
}

}  // namespace

ShadowMapPass::Outputs ShadowMapPass::add_to_frame_graph(FrameGraph& fg, const PassContext& ctx,
                                                         const Inputs&) {
    PTS_ZONE_SCOPED;
    ensure_initialized(ctx.device);

    auto desc_layout = fg.bind_group_layout(
        "shadow_map/desc", shadow_shader::create_bind_group_layout_0(ctx.device.handle()));
    // Consumer layout is registered up-front by the owning renderer (e.g. forwardPass)
    // using its own shader's reflection, since the shape of the consumer-side
    // descriptor is a property of how downstream passes read shadow output, not of
    // shadow.slang.
    auto consumer_bgl = fg.bind_group_layout("shadow_map/consumer");

    // Position-only vertex layout: stride=12, one Float32x3 at offset 0, location 0
    WGPUVertexAttribute pos_attr{};
    pos_attr.format = WGPUVertexFormat_Float32x3;
    pos_attr.offset = 0;
    pos_attr.shaderLocation = 0;

    auto* pipeline_handle = fg.render_pipeline("shadow_map")
                                .shader("core/generated/shaders/shadow.wgsl")
                                .no_fragment()
                                .depth_format(WGPUTextureFormat_Depth32Float)
                                .depth_write(true)
                                .depth_compare(WGPUCompareFunction_Less)
                                .cull_mode(WGPUCullMode_Front)
                                .depth_bias(0, 0.0f)
                                .vertex_buffer({12, WGPUVertexStepMode_Vertex, {pos_attr}})
                                .bind_group_layouts({desc_layout})
                                .build();

    // Count shadow-casting lights (distant + rect/disk area lights).
    auto lights = ctx.world.get_lights().span_raw();
    uint32_t shadow_count = 0;
    if (m_enabled) {
        for (uint32_t li = 0; li < static_cast<uint32_t>(lights.size()); ++li) {
            if (!lights[li].active) continue;
            if (!casts_shadow_map(lights[li].value.type)) continue;
            if (!lights[li].value.casts_shadow) continue;
            ++shadow_count;
            if (shadow_count >= k_max_shadow_maps) break;
        }
    }

    // Always ensure at least 1 layer for downstream descriptors
    uint32_t layer_count = std::max(shadow_count, 1u);

    // Register shadow texture array with frame graph
    TextureDesc shadow_tex_desc;
    shadow_tex_desc.width = m_resolution;
    shadow_tex_desc.height = m_resolution;
    shadow_tex_desc.array_layers = layer_count;
    shadow_tex_desc.format = WGPUTextureFormat_Depth32Float;
    shadow_tex_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                                          WGPUTextureUsage_TextureBinding);
    shadow_tex_desc.force_array_view = true;
    auto shadow_array = create_texture(fg, shadow_tex_desc, "shadow_depth_array");

    // Register shadow info buffer (one ShadowInfo per light, minimum 1)
    auto info_count = std::max(uint32_t(1), static_cast<uint32_t>(lights.size()));
    uint64_t info_bytes = static_cast<uint64_t>(info_count) * sizeof(ShadowInfo);
    BufferDesc info_buf_desc;
    info_buf_desc.size = info_bytes;
    info_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    auto shadow_info_buf = create_buffer(fg, info_buf_desc, "shadow_info");

    if (shadow_count == 0) {
        // Upload all-inactive ShadowInfo entries
        auto queue = ctx.queue;
        auto empty_infos = std::vector<ShadowInfo>(info_count);
        fg.add_pass("shadow_info_upload")
            .execute([queue, shadow_info_buf, infos = std::move(empty_infos)](
                         ExecuteContext& exec, WGPUComputePassEncoder) {
                auto buf = exec.get(shadow_info_buf).buffer;
                wgpuQueueWriteBuffer(queue, buf, 0, infos.data(),
                                     infos.size() * sizeof(ShadowInfo));
            });
        auto consumer = descriptor(fg, consumer_bgl, "consumer_desc")
                            .buffer(0, shadow_info_buf)
                            .texture(1, shadow_array)
                            .sampler(2, fg.sampler(WGPUSamplerBindingType_NonFiltering))
                            .build();
        return {shadow_array, shadow_info_buf, consumer};
    }

    // Scene AABB from TLAS root (built by RenderWorld::prepare_gpu_buffers)
    auto scene_bounds = ctx.world.scene_bounds();
    auto aabb_min = scene_bounds.min;
    auto aabb_max = scene_bounds.max;

    auto objects = ctx.world.get_objects().span_raw();
    uint32_t total_slots = static_cast<uint32_t>(objects.size());

    // Build one ShadowInfo per light (matching light buffer order)
    std::vector<ShadowInfo> infos(lights.size());
    uint32_t layer_index = 0;

    for (uint32_t li = 0; li < static_cast<uint32_t>(lights.size()); ++li) {
        if (!lights[li].active) continue;
        const auto& light = lights[li].value;
        if (!casts_shadow_map(light.type)) continue;
        if (!light.casts_shadow) continue;
        if (layer_index >= k_max_shadow_maps) continue;

        LightProjection proj = (light.type == LightData::Type::Distant)
                                   ? compute_distant_light_vp(light, aabb_min, aabb_max)
                                   : compute_area_light_vp(light, aabb_min, aabb_max);

        infos[li].light_vp = proj.vp;
        infos[li].texel_size = 1.0f / static_cast<float>(m_resolution);
        infos[li].normal_bias = 0.0f;
        infos[li].has_shadow = 1;
        infos[li].layer = layer_index;
        infos[li].light_near = proj.near_plane;
        infos[li].light_far = proj.far_plane;
        infos[li].light_size_uv = proj.light_size_uv;
        infos[li].projection_type = proj.projection_type;
        ++layer_index;
    }
    INVARIANT(layer_index == shadow_count);

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
        std::max(uint64_t(1), static_cast<uint64_t>(layer_index)) * k_uniform_align;
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

    // Extract per-layer view-projection matrices
    std::vector<glm::mat4> layer_vps;
    layer_vps.reserve(layer_index);
    for (uint32_t li = 0; li < static_cast<uint32_t>(lights.size()); ++li) {
        if (infos[li].has_shadow) layer_vps.push_back(infos[li].light_vp);
    }
    INVARIANT(layer_vps.size() == layer_index);

    auto queue = ctx.queue;
    const auto& world = ctx.world;

    // Upload shadow info + model matrices + light VPs in a single compute pass
    fg.add_pass("shadow_upload")
        .execute([queue, shadow_info_buf, model_buf_decl, vp_buf_decl, layer_index,
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
            for (uint32_t l = 0; l < layer_index; ++l) {
                wgpuQueueWriteBuffer(queue, vp_buf, l * k_uniform_align, &layer_vps[l],
                                     sizeof(glm::mat4));
            }
        });

    // Render each shadow layer
    for (uint32_t layer = 0; layer < layer_index; ++layer) {
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

    auto consumer = descriptor(fg, consumer_bgl, "consumer_desc")
                        .buffer(0, shadow_info_buf)
                        .texture(1, shadow_array)
                        .sampler(2, fg.sampler(WGPUSamplerBindingType_NonFiltering))
                        .build();
    return {shadow_array, shadow_info_buf, consumer};
}

void ShadowMapPass::draw_imgui() {
    ImGui::Checkbox("Enabled", &m_enabled);
}

}  // namespace pts::rendering
