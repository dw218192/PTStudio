#include <core/diagnostics.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/webgpu/pipelineBuilder.h>

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace pts::rendering {

ShadowMapPass::ShadowMapPass(const ShaderLoader& sl) : IRenderPass(sl) {
}

ShadowMapPass::~ShadowMapPass() {
    release_shadow_texture();
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group) wgpuBindGroupRelease(ready->bind_group);
        if (ready->bgl) wgpuBindGroupLayoutRelease(ready->bgl);
    }
}

auto ShadowMapPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

WGPUTextureView ShadowMapPass::shadow_array_view() const {
    return m_shadow_array_view;
}

void ShadowMapPass::do_setup(const webgpu::Device& device) {
    // Release existing state for re-entry (hot-reload)
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group) wgpuBindGroupRelease(ready->bind_group);
        if (ready->bgl) wgpuBindGroupLayoutRelease(ready->bgl);
    }

    auto shader_src = get_shader_loader().load("core/generated/shaders/shadow.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    // BGL: binding 0 = uniform (dynamic), one mat4 (64 bytes)
    WGPUBindGroupLayoutEntry bgl_entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    bgl_entry.binding = 0;
    bgl_entry.visibility = WGPUShaderStage_Vertex;
    bgl_entry.buffer.type = WGPUBufferBindingType_Uniform;
    bgl_entry.buffer.hasDynamicOffset = true;
    bgl_entry.buffer.minBindingSize = 64;  // one mat4

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = 1;
    bgl_desc.entries = &bgl_entry;
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

    uint32_t initial_capacity = 64;
    auto uniform_buf = device.create_buffer(
        k_uniform_align * initial_capacity,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    // Create bind group
    WGPUBindGroupEntry bg_entry = WGPU_BIND_GROUP_ENTRY_INIT;
    bg_entry.binding = 0;
    bg_entry.buffer = uniform_buf.handle();
    bg_entry.offset = 0;
    bg_entry.size = 64;  // one mat4

    WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg_desc.layout = bgl;
    bg_desc.entryCount = 1;
    bg_desc.entries = &bg_entry;
    auto bind_group = wgpuDeviceCreateBindGroup(device.handle(), &bg_desc);

    m_state = Ready{
        std::move(shader), std::move(pipeline), std::move(uniform_buf), bgl,
        bind_group,        initial_capacity,
    };
}

void ShadowMapPass::add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) {
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    // Count shadow-casting distant lights
    auto lights = ctx.world.get_lights();
    uint32_t shadow_count = 0;
    for (uint32_t li = 0; li < static_cast<uint32_t>(lights.size()); ++li) {
        if (!lights[li].active()) continue;
        if (lights[li]->type != LightData::Type::Distant) continue;
        if (!lights[li]->casts_shadow) continue;
        ++shadow_count;
        if (shadow_count >= k_max_shadow_maps) break;
    }

    // Always ensure a valid texture (at least 1 layer) for downstream bind groups
    ensure_shadow_texture(ctx.device, std::max(shadow_count, 1u));

    if (shadow_count == 0) {
        // Write all-inactive ShadowInfo entries so the buffer is valid
        std::vector<ShadowInfo> empty(std::max(1u, static_cast<uint32_t>(lights.size())));
        ctx.world.set_shadow_data(empty, ctx.device, ctx.queue);
        return;
    }

    // Scene AABB from BVH root node (built by RenderWorld::prepare_gpu_buffers)
    auto scene_bounds = ctx.world.scene_bvh().scene_bounds();
    auto aabb_min = scene_bounds.min;
    auto aabb_max = scene_bounds.max;

    auto objects = ctx.world.get_objects();

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

    // Upload shadow data to RenderWorld
    ctx.world.set_shadow_data(infos, ctx.device, ctx.queue);

    // Ensure texture array
    ensure_shadow_texture(ctx.device, layer_index);

    // Count active objects
    uint32_t active_object_count = 0;
    for (uint32_t oi = 0; oi < static_cast<uint32_t>(objects.size()); ++oi) {
        if (objects[oi].active()) ++active_object_count;
    }

    // Resize uniform buffer if needed: layers * total_slots * k_uniform_align
    uint32_t total_slots = static_cast<uint32_t>(objects.size());
    uint64_t needed_size = static_cast<uint64_t>(layer_index) * total_slots * k_uniform_align;
    if (needed_size > 0 &&
        needed_size > static_cast<uint64_t>(ready.object_capacity) * k_uniform_align) {
        uint32_t new_capacity = ready.object_capacity;
        uint32_t needed_capacity =
            static_cast<uint32_t>(static_cast<uint64_t>(layer_index) * total_slots);
        while (new_capacity < needed_capacity) {
            new_capacity *= 2;
        }

        ready.per_object_uniform_buf = ctx.device.create_buffer(
            static_cast<uint64_t>(new_capacity) * k_uniform_align,
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));
        ready.object_capacity = new_capacity;

        // Recreate bind group
        if (ready.bind_group) wgpuBindGroupRelease(ready.bind_group);

        WGPUBindGroupEntry bg_entry = WGPU_BIND_GROUP_ENTRY_INIT;
        bg_entry.binding = 0;
        bg_entry.buffer = ready.per_object_uniform_buf.handle();
        bg_entry.offset = 0;
        bg_entry.size = 64;

        WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bg_desc.layout = ready.bgl;
        bg_desc.entryCount = 1;
        bg_desc.entries = &bg_entry;
        ready.bind_group = wgpuDeviceCreateBindGroup(ctx.device.handle(), &bg_desc);
    }

    // Build layer → light index mapping
    std::vector<uint32_t> layer_to_light;
    layer_to_light.reserve(layer_index);
    for (uint32_t li = 0; li < static_cast<uint32_t>(lights.size()); ++li) {
        if (infos[li].has_shadow) layer_to_light.push_back(li);
    }
    INVARIANT(layer_to_light.size() == layer_index);

    // For each shadow layer, upload uniforms and add a frame graph pass
    auto* pipeline_handle = ready.pipeline.handle();
    auto bind_group = ready.bind_group;
    auto uniform_buf = ready.per_object_uniform_buf.handle();
    const auto& world = ctx.world;

    for (uint32_t layer = 0; layer < layer_index; ++layer) {
        const auto& light_vp = infos[layer_to_light[layer]].light_vp;

        // Write per-object uniforms for this layer
        // Interleaved: buffer[layer * total_slots + obj_index]
        for (uint32_t oi = 0; oi < total_slots; ++oi) {
            if (!objects[oi].active()) continue;
            glm::mat4 light_mvp = light_vp * objects[oi]->transform;
            uint64_t offset = (static_cast<uint64_t>(layer) * total_slots + oi) * k_uniform_align;
            wgpuQueueWriteBuffer(ctx.queue, uniform_buf, offset, &light_mvp, sizeof(glm::mat4));
        }

        // Emit frame graph pass
        auto layer_val = layer;
        fg.add_pass("shadow_depth_" + std::to_string(layer))
            .depth(m_shadow_layer_views[layer], 1.0f)
            .execute([=, &world](WGPURenderPassEncoder pass) {
                auto objs = world.get_objects();
                auto mesh_slots = world.get_meshes();
                wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
                for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                    if (!objs[i].active()) continue;
                    uint32_t dyn_offset = static_cast<uint32_t>(
                        (static_cast<uint64_t>(layer_val) * objs.size() + i) * k_uniform_align);
                    wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 1, &dyn_offset);
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
}

void ShadowMapPass::ensure_shadow_texture(const webgpu::Device& device, uint32_t layer_count) {
    PRECONDITION(layer_count > 0);
    if (layer_count == m_current_layer_count && m_shadow_texture) return;
    release_shadow_texture();

    WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    desc.size = {m_resolution, m_resolution, layer_count};
    desc.format = WGPUTextureFormat_Depth32Float;
    desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment |
                                               WGPUTextureUsage_TextureBinding);
    desc.mipLevelCount = 1;
    desc.sampleCount = 1;
    desc.dimension = WGPUTextureDimension_2D;
    m_shadow_texture = wgpuDeviceCreateTexture(device.handle(), &desc);

    // Full array view for sampling in shaders
    WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    view_desc.format = WGPUTextureFormat_Depth32Float;
    view_desc.dimension = WGPUTextureViewDimension_2DArray;
    view_desc.arrayLayerCount = layer_count;
    view_desc.mipLevelCount = 1;
    m_shadow_array_view = wgpuTextureCreateView(m_shadow_texture, &view_desc);

    // Per-layer views for rendering
    m_shadow_layer_views.resize(layer_count);
    for (uint32_t i = 0; i < layer_count; ++i) {
        WGPUTextureViewDescriptor lv = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
        lv.format = WGPUTextureFormat_Depth32Float;
        lv.dimension = WGPUTextureViewDimension_2D;
        lv.baseArrayLayer = i;
        lv.arrayLayerCount = 1;
        lv.mipLevelCount = 1;
        m_shadow_layer_views[i] = wgpuTextureCreateView(m_shadow_texture, &lv);
    }
    m_current_layer_count = layer_count;
}

void ShadowMapPass::release_shadow_texture() {
    for (auto view : m_shadow_layer_views) {
        if (view) wgpuTextureViewRelease(view);
    }
    m_shadow_layer_views.clear();
    if (m_shadow_array_view) {
        wgpuTextureViewRelease(m_shadow_array_view);
        m_shadow_array_view = nullptr;
    }
    if (m_shadow_texture) {
        wgpuTextureDestroy(m_shadow_texture);
        wgpuTextureRelease(m_shadow_texture);
        m_shadow_texture = nullptr;
    }
    m_current_layer_count = 0;
}

}  // namespace pts::rendering
