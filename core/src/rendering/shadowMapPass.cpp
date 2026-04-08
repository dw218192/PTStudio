#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace pts::rendering {

ShadowMapPass::ShadowMapPass(const ShaderLoader& sl) : IPass(sl) {
}

ShadowMapPass::~ShadowMapPass() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bgl) wgpuBindGroupLayoutRelease(ready->bgl);
    }
}

auto ShadowMapPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

void ShadowMapPass::do_setup(const webgpu::Device& device) {
    // Release existing state for re-entry (hot-reload)
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bgl) wgpuBindGroupLayoutRelease(ready->bgl);
    }

    auto shader_src = get_shader_loader().load("core/generated/shaders/shadow.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    // BGL: binding 0 = model matrix (dynamic), binding 1 = light VP (dynamic)
    WGPUBindGroupLayoutEntry bgl_entries[2] = {};
    bgl_entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    bgl_entries[0].binding = 0;
    bgl_entries[0].visibility = WGPUShaderStage_Vertex;
    bgl_entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    bgl_entries[0].buffer.hasDynamicOffset = true;
    bgl_entries[0].buffer.minBindingSize = 64;  // one mat4 (model)

    bgl_entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    bgl_entries[1].binding = 1;
    bgl_entries[1].visibility = WGPUShaderStage_Vertex;
    bgl_entries[1].buffer.type = WGPUBufferBindingType_Uniform;
    bgl_entries[1].buffer.hasDynamicOffset = true;
    bgl_entries[1].buffer.minBindingSize = 64;  // one mat4 (light VP)

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = 2;
    bgl_desc.entries = bgl_entries;
    auto bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);

    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &bgl;
    auto pipeline_layout = wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

    // Position-only vertex layout: stride=12, one Float32x3 at offset 0, location 0
    webgpu::VertexBufferLayout vtx_layout;
    vtx_layout.stride = 12;
    vtx_layout.step_mode = WGPUVertexStepMode_Vertex;
    WGPUVertexAttribute pos_attr{};
    pos_attr.format = WGPUVertexFormat_Float32x3;
    pos_attr.offset = 0;
    pos_attr.shaderLocation = 0;
    vtx_layout.attributes.push_back(pos_attr);

    auto pipeline = webgpu::RenderPipelineBuilder(device)
                        .shader(shader)
                        .no_fragment()
                        .depth_format(WGPUTextureFormat_Depth32Float)
                        .depth_write(true)
                        .depth_compare(WGPUCompareFunction_Less)
                        .cull_mode(WGPUCullMode_Front)
                        .depth_bias(0, 0.0f)
                        .vertex_buffer(std::move(vtx_layout))
                        .pipeline_layout(pipeline_layout)
                        .build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    m_state = Ready{
        std::move(shader),
        std::move(pipeline),
        bgl,
    };
}

ShadowMapPass::Outputs ShadowMapPass::add_to_frame_graph(FrameGraph& fg, const PassContext& ctx,
                                                         const Inputs&) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    // Count shadow-casting distant lights
    auto lights = ctx.world.get_lights();
    uint32_t shadow_count = 0;
    if (m_enabled) {
        for (uint32_t li = 0; li < static_cast<uint32_t>(lights.size()); ++li) {
            if (!lights[li].active()) continue;
            if (lights[li]->type != LightData::Type::Distant) continue;
            if (!lights[li]->casts_shadow) continue;
            ++shadow_count;
            if (shadow_count >= k_max_shadow_maps) break;
        }
    }

    // Always ensure at least 1 layer for downstream bind groups
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
            .execute([queue, shadow_info_buf, infos = std::move(empty_infos),
                      &fg](WGPUComputePassEncoder) {
                auto buf = fg.get_buffer_ref(shadow_info_buf).handle();
                wgpuQueueWriteBuffer(queue, buf, 0, infos.data(),
                                     infos.size() * sizeof(ShadowInfo));
            });
        return {shadow_array, shadow_info_buf};
    }

    // Scene AABB from TLAS root (built by RenderWorld::prepare_gpu_buffers)
    auto scene_bounds = ctx.world.scene_bounds();
    auto aabb_min = scene_bounds.min;
    auto aabb_max = scene_bounds.max;

    auto objects = ctx.world.get_objects();
    uint32_t total_slots = static_cast<uint32_t>(objects.size());

    // Build one ShadowInfo per light (matching light buffer order)
    std::vector<ShadowInfo> infos(lights.size());
    uint32_t layer_index = 0;

    for (uint32_t li = 0; li < static_cast<uint32_t>(lights.size()); ++li) {
        if (!lights[li].active()) continue;
        if (lights[li]->type != LightData::Type::Distant) continue;
        if (!lights[li]->casts_shadow) continue;
        if (layer_index >= k_max_shadow_maps) continue;

        auto dir = glm::normalize(lights[li]->direction);

        auto center = (aabb_min + aabb_max) * 0.5f;
        auto half_diag = glm::length(aabb_max - aabb_min) * 0.5f;

        // Choose up vector that isn't parallel to direction
        auto up = glm::vec3(0, 1, 0);
        if (std::abs(glm::dot(dir, up)) > 0.99f) up = glm::vec3(1, 0, 0);

        auto light_view = glm::lookAt(center - dir * half_diag, center, up);

        // Transform all 8 AABB corners into light space to find bounds
        glm::vec3 ls_min(std::numeric_limits<float>::max());
        glm::vec3 ls_max(std::numeric_limits<float>::lowest());
        for (int c = 0; c < 8; ++c) {
            glm::vec3 corner((c & 1) ? aabb_max.x : aabb_min.x, (c & 2) ? aabb_max.y : aabb_min.y,
                             (c & 4) ? aabb_max.z : aabb_min.z);
            glm::vec3 ls_pt = glm::vec3(light_view * glm::vec4(corner, 1.0f));
            ls_min = glm::min(ls_min, ls_pt);
            ls_max = glm::max(ls_max, ls_pt);
        }

        auto ortho_proj = glm::ortho(ls_min.x, ls_max.x, ls_min.y, ls_max.y, -ls_max.z, -ls_min.z);

        infos[li].light_vp = ortho_proj * light_view;
        infos[li].texel_size = 1.0f / static_cast<float>(m_resolution);
        infos[li].normal_bias = 0.0f;
        infos[li].has_shadow = 1;
        infos[li].layer = layer_index;
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
    auto model_buf_handle = create_buffer(fg, model_buf_desc, "models");

    // Light VP buffer: one VP matrix per shadow layer
    uint64_t vp_buf_size =
        std::max(uint64_t(1), static_cast<uint64_t>(layer_index)) * k_uniform_align;
    BufferDesc vp_buf_desc;
    vp_buf_desc.size = vp_buf_size;
    vp_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto vp_buf_handle = create_buffer(fg, vp_buf_desc, "light_vps");

    // Bind group: binding 0 = model (dynamic), binding 1 = light VP (dynamic)
    BindGroupDesc bg_desc;
    bg_desc.layout = ready.bgl;
    bg_desc.entries = {
        {0, ManagedBufferBinding{model_buf_handle, 0, 64}},
        {1, ManagedBufferBinding{vp_buf_handle, 0, 64}},
    };
    auto bg_handle = create_bind_group(fg, std::move(bg_desc), "bg0");

    // Extract per-layer view-projection matrices
    std::vector<glm::mat4> layer_vps;
    layer_vps.reserve(layer_index);
    for (uint32_t li = 0; li < static_cast<uint32_t>(lights.size()); ++li) {
        if (infos[li].has_shadow) layer_vps.push_back(infos[li].light_vp);
    }
    INVARIANT(layer_vps.size() == layer_index);

    auto* pipeline_handle = ready.pipeline.handle();
    auto queue = ctx.queue;
    const auto& world = ctx.world;

    // Upload shadow info + model matrices + light VPs in a single compute pass
    fg.add_pass("shadow_upload")
        .execute([queue, shadow_info_buf, model_buf_handle, vp_buf_handle, layer_index,
                  infos = std::move(infos), layer_vps = std::move(layer_vps), &fg,
                  &world](WGPUComputePassEncoder) {
            // Shadow info buffer
            auto info_buf = fg.get_buffer_ref(shadow_info_buf).handle();
            wgpuQueueWriteBuffer(queue, info_buf, 0, infos.data(),
                                 infos.size() * sizeof(ShadowInfo));

            // Model matrices (uploaded once, shared across all layers)
            auto model_buf = fg.get_buffer_ref(model_buf_handle).handle();
            auto objs = world.get_objects();
            for (uint32_t oi = 0; oi < static_cast<uint32_t>(objs.size()); ++oi) {
                if (!objs[oi].active()) continue;
                if (!objs[oi]->visible) continue;
                wgpuQueueWriteBuffer(queue, model_buf, oi * k_uniform_align, &objs[oi]->transform,
                                     sizeof(glm::mat4));
            }

            // Light VP matrices
            auto vp_buf = fg.get_buffer_ref(vp_buf_handle).handle();
            for (uint32_t l = 0; l < layer_index; ++l) {
                wgpuQueueWriteBuffer(queue, vp_buf, l * k_uniform_align, &layer_vps[l],
                                     sizeof(glm::mat4));
            }
        });

    // Render each shadow layer
    for (uint32_t layer = 0; layer < layer_index; ++layer) {
        fg.add_pass("shadow_depth_" + std::to_string(layer))
            .depth(shadow_array, layer)
            .execute([=, &fg, &world](WGPURenderPassEncoder pass) {
                auto bg = fg.get_bind_group_ref(bg_handle).handle();
                auto objs = world.get_objects();
                auto mesh_slots = world.get_meshes();
                uint32_t slots = static_cast<uint32_t>(objs.size());

                uint32_t vp_offset = layer * k_uniform_align;
                wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
                for (uint32_t i = 0; i < slots; ++i) {
                    if (!objs[i].active()) continue;
                    if (!objs[i]->visible) continue;
                    uint32_t model_offset = i * k_uniform_align;
                    uint32_t dyn_offsets[2] = {model_offset, vp_offset};
                    wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 2, dyn_offsets);
                    const auto& mesh = mesh_slots[objs[i]->mesh_index];
                    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh->position_buffer.handle(), 0,
                                                         mesh->position_buffer.size());
                    wgpuRenderPassEncoderSetIndexBuffer(pass, mesh->index_buffer.handle(),
                                                        WGPUIndexFormat_Uint32, 0,
                                                        mesh->index_buffer.size());
                    wgpuRenderPassEncoderDrawIndexed(pass, mesh->index_count, 1, 0, 0, 0);
                }
            });
    }

    return {shadow_array, shadow_info_buf};
}

void ShadowMapPass::draw_imgui() {
    ImGui::Checkbox("Enabled", &m_enabled);
}

}  // namespace pts::rendering
