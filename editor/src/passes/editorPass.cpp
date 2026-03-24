#include "editorPass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <gizmo_shader_metadata.h>
#include <picking_shader_metadata.h>

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

using namespace pts;
using namespace pts::editor;

// ── Uniform structs ────────────────────────────────────────────────────

struct PickingUniforms {
    glm::mat4 mvp;
    uint32_t object_id;
    uint32_t _pad[3];
};
static_assert(sizeof(PickingUniforms) == 80);

struct GizmoUniforms {
    glm::mat4 mvp;
    glm::vec4 color;
};
static_assert(sizeof(GizmoUniforms) == 80);

static_assert(EditorPass::k_uniform_align >= sizeof(PickingUniforms));
static_assert(EditorPass::k_uniform_align >= sizeof(GizmoUniforms));

// ── Bind group helper ──────────────────────────────────────────────────

static WGPUBindGroup create_bind_group(WGPUDevice device, WGPUBindGroupLayout layout,
                                       WGPUBuffer uniform_buf, uint64_t min_binding_size) {
    WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
    entry.binding = 0;
    entry.buffer = uniform_buf;
    entry.offset = 0;
    entry.size = min_binding_size;

    WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg_desc.layout = layout;
    bg_desc.entryCount = 1;
    bg_desc.entries = &entry;
    return wgpuDeviceCreateBindGroup(device, &bg_desc);
}

// ── Gizmo geometry generation ──────────────────────────────────────────

static constexpr uint32_t k_circle_segments = 48;

static void generate_circle(std::vector<glm::vec3>& out, glm::vec3 center, glm::vec3 axis_a,
                            glm::vec3 axis_b, float radius) {
    for (uint32_t i = 0; i < k_circle_segments; ++i) {
        float a0 = glm::two_pi<float>() * static_cast<float>(i) / k_circle_segments;
        float a1 = glm::two_pi<float>() * static_cast<float>(i + 1) / k_circle_segments;
        out.push_back(center + (std::cos(a0) * axis_a + std::sin(a0) * axis_b) * radius);
        out.push_back(center + (std::cos(a1) * axis_a + std::sin(a1) * axis_b) * radius);
    }
}

/// Generate filled triangle-list geometry for the picking pass.
static std::vector<glm::vec3> generate_light_pick_verts(const rendering::LightData& light) {
    std::vector<glm::vec3> verts;
    auto triangle_fan = [&](glm::vec3 center, glm::vec3 axis_a, glm::vec3 axis_b, float radius) {
        for (uint32_t i = 0; i < k_circle_segments; ++i) {
            float a0 = glm::two_pi<float>() * static_cast<float>(i) / k_circle_segments;
            float a1 = glm::two_pi<float>() * static_cast<float>(i + 1) / k_circle_segments;
            verts.push_back(center);
            verts.push_back(center + (std::cos(a0) * axis_a + std::sin(a0) * axis_b) * radius);
            verts.push_back(center + (std::cos(a1) * axis_a + std::sin(a1) * axis_b) * radius);
        }
    };
    switch (light.type) {
        case rendering::LightData::Type::Sphere: {
            float r = std::max(light.radius, 0.1f);
            verts.reserve(k_circle_segments * 3 * 3);
            triangle_fan({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, r);
            triangle_fan({0, 0, 0}, {1, 0, 0}, {0, 0, 1}, r);
            triangle_fan({0, 0, 0}, {0, 1, 0}, {0, 0, 1}, r);
            break;
        }
        case rendering::LightData::Type::Rect: {
            float hw = light.width * 0.5f;
            float hh = light.height * 0.5f;
            verts = {{-hw, -hh, 0}, {hw, -hh, 0}, {hw, hh, 0},
                     {-hw, -hh, 0}, {hw, hh, 0},  {-hw, hh, 0}};
            break;
        }
        case rendering::LightData::Type::Disk: {
            float r = std::max(light.radius, 0.1f);
            verts.reserve(k_circle_segments * 3);
            triangle_fan({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, r);
            break;
        }
        case rendering::LightData::Type::Distant:
        case rendering::LightData::Type::Dome:
            break;
    }
    return verts;
}

/// Generate line-list wireframe geometry for the color overlay.
static std::vector<glm::vec3> generate_light_verts(const rendering::LightData& light) {
    std::vector<glm::vec3> verts;
    switch (light.type) {
        case rendering::LightData::Type::Sphere: {
            float r = std::max(light.radius, 0.1f);
            verts.reserve(k_circle_segments * 2 * 3);
            generate_circle(verts, {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, r);
            generate_circle(verts, {0, 0, 0}, {1, 0, 0}, {0, 0, 1}, r);
            generate_circle(verts, {0, 0, 0}, {0, 1, 0}, {0, 0, 1}, r);
            break;
        }
        case rendering::LightData::Type::Rect: {
            float hw = light.width * 0.5f;
            float hh = light.height * 0.5f;
            float arrow = std::min(hw, hh) * 0.7f;
            verts = {{-hw, -hh, 0},
                     {hw, -hh, 0},
                     {hw, -hh, 0},
                     {hw, hh, 0},
                     {hw, hh, 0},
                     {-hw, hh, 0},
                     {-hw, hh, 0},
                     {-hw, -hh, 0},
                     // Direction arrow along -Z (emission direction)
                     {0, 0, 0},
                     {0, 0, -arrow}};
            break;
        }
        case rendering::LightData::Type::Disk: {
            float r = std::max(light.radius, 0.1f);
            float arrow = r * 0.7f;
            verts.reserve(k_circle_segments * 2 + 2);
            generate_circle(verts, {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, r);
            // Direction arrow along -Z (emission direction)
            verts.push_back({0, 0, 0});
            verts.push_back({0, 0, -arrow});
            break;
        }
        case rendering::LightData::Type::Distant:
        case rendering::LightData::Type::Dome:
            break;
    }
    return verts;
}

// ── EditorPass implementation ──────────────────────────────────────────

EditorPass::~EditorPass() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->picking_bind_group) wgpuBindGroupRelease(ready->picking_bind_group);
        if (ready->picking_bind_group_layout)
            wgpuBindGroupLayoutRelease(ready->picking_bind_group_layout);
        if (ready->gizmo_bind_group) wgpuBindGroupRelease(ready->gizmo_bind_group);
        if (ready->gizmo_bind_group_layout)
            wgpuBindGroupLayoutRelease(ready->gizmo_bind_group_layout);
    }
}

auto EditorPass::name() const noexcept -> std::string_view {
    return "editor";
}

auto EditorPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

void EditorPass::do_setup(const webgpu::Device& device) {
    WGPUBindGroup old_picking_bg = nullptr, old_gizmo_bg = nullptr;
    WGPUBindGroupLayout old_picking_bgl = nullptr, old_gizmo_bgl = nullptr;
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        old_picking_bg = ready->picking_bind_group;
        old_picking_bgl = ready->picking_bind_group_layout;
        old_gizmo_bg = ready->gizmo_bind_group;
        old_gizmo_bgl = ready->gizmo_bind_group_layout;
        ready->picking_bind_group = nullptr;
        ready->picking_bind_group_layout = nullptr;
        ready->gizmo_bind_group = nullptr;
        ready->gizmo_bind_group_layout = nullptr;
    }

    uint32_t initial_capacity = 64;

    // ── Picking pipeline (mesh objects + light shapes) ─────────────────
    auto picking_src = get_shader_loader().load("editor/generated/shaders/picking.wgsl");
    auto picking_shader = device.create_shader_module_from_source(picking_src);

    auto picking_uniform_buffer = device.create_buffer(
        k_uniform_align * initial_capacity,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    WGPUBindGroupLayoutEntry picking_bgl_entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    picking_bgl_entry.binding = 0;
    picking_bgl_entry.visibility =
        static_cast<WGPUShaderStage>(WGPUShaderStage_Vertex | WGPUShaderStage_Fragment);
    picking_bgl_entry.buffer.type = WGPUBufferBindingType_Uniform;
    picking_bgl_entry.buffer.hasDynamicOffset = true;
    picking_bgl_entry.buffer.minBindingSize = sizeof(PickingUniforms);

    WGPUBindGroupLayoutDescriptor picking_bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    picking_bgl_desc.entryCount = 1;
    picking_bgl_desc.entries = &picking_bgl_entry;
    auto picking_bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &picking_bgl_desc);

    auto picking_bg = create_bind_group(device.handle(), picking_bgl,
                                        picking_uniform_buffer.handle(), sizeof(PickingUniforms));

    WGPUPipelineLayoutDescriptor picking_pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    picking_pl_desc.bindGroupLayoutCount = 1;
    picking_pl_desc.bindGroupLayouts = &picking_bgl;
    auto picking_pl = wgpuDeviceCreatePipelineLayout(device.handle(), &picking_pl_desc);

    auto picking_pipeline = webgpu::RenderPipelineBuilder(device)
                                .shader(picking_shader)
                                .color_format(WGPUTextureFormat_R32Uint)
                                .depth_format(WGPUTextureFormat_Depth24Plus)
                                .depth_write(true)
                                .depth_compare(WGPUCompareFunction_Less)
                                .cull_mode(WGPUCullMode_Back)
                                .pipeline_layout(picking_pl)
                                .vertex_layout<editor_picking_shader::VertexLayout>()
                                .build();

    // Picking pipeline variant for gizmo lines (LineList, no cull)
    auto gizmo_picking_pipeline = webgpu::RenderPipelineBuilder(device)
                                      .shader(picking_shader)
                                      .color_format(WGPUTextureFormat_R32Uint)
                                      .depth_format(WGPUTextureFormat_Depth24Plus)
                                      .depth_write(true)
                                      .depth_compare(WGPUCompareFunction_Less)
                                      .cull_mode(WGPUCullMode_None)
                                      .topology(WGPUPrimitiveTopology_TriangleList)
                                      .pipeline_layout(picking_pl)
                                      .vertex_layout<editor_gizmo_shader::VertexLayout>()
                                      .build();

    wgpuPipelineLayoutRelease(picking_pl);

    // ── Gizmo color pipeline (wireframe overlay on scene_color) ────────
    auto gizmo_src = get_shader_loader().load("editor/generated/shaders/gizmo.wgsl");
    auto gizmo_shader = device.create_shader_module_from_source(gizmo_src);

    auto gizmo_uniform_buffer = device.create_buffer(
        k_uniform_align * initial_capacity,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    WGPUBindGroupLayoutEntry gizmo_bgl_entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    gizmo_bgl_entry.binding = 0;
    gizmo_bgl_entry.visibility =
        static_cast<WGPUShaderStage>(WGPUShaderStage_Vertex | WGPUShaderStage_Fragment);
    gizmo_bgl_entry.buffer.type = WGPUBufferBindingType_Uniform;
    gizmo_bgl_entry.buffer.hasDynamicOffset = true;
    gizmo_bgl_entry.buffer.minBindingSize = sizeof(GizmoUniforms);

    WGPUBindGroupLayoutDescriptor gizmo_bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    gizmo_bgl_desc.entryCount = 1;
    gizmo_bgl_desc.entries = &gizmo_bgl_entry;
    auto gizmo_bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &gizmo_bgl_desc);

    auto gizmo_bg = create_bind_group(device.handle(), gizmo_bgl, gizmo_uniform_buffer.handle(),
                                      sizeof(GizmoUniforms));

    WGPUPipelineLayoutDescriptor gizmo_pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    gizmo_pl_desc.bindGroupLayoutCount = 1;
    gizmo_pl_desc.bindGroupLayouts = &gizmo_bgl;
    auto gizmo_pl = wgpuDeviceCreatePipelineLayout(device.handle(), &gizmo_pl_desc);

    WGPUBlendState blend = {};
    blend.color.operation = WGPUBlendOperation_Add;
    blend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blend.alpha.operation = WGPUBlendOperation_Add;
    blend.alpha.srcFactor = WGPUBlendFactor_One;
    blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;

    auto gizmo_color_pipeline = webgpu::RenderPipelineBuilder(device)
                                    .shader(gizmo_shader)
                                    .color_format(WGPUTextureFormat_RGBA8Unorm)
                                    .blend_state(blend)
                                    .cull_mode(WGPUCullMode_None)
                                    .topology(WGPUPrimitiveTopology_LineList)
                                    .pipeline_layout(gizmo_pl)
                                    .vertex_layout<editor_gizmo_shader::VertexLayout>()
                                    .build();

    wgpuPipelineLayoutRelease(gizmo_pl);

    m_state = Ready{
        std::move(picking_shader),
        std::move(picking_pipeline),
        std::move(picking_uniform_buffer),
        picking_bg,
        picking_bgl,
        initial_capacity,

        std::move(gizmo_shader),
        std::move(gizmo_color_pipeline),
        std::move(gizmo_picking_pipeline),
        std::move(gizmo_uniform_buffer),
        gizmo_bg,
        gizmo_bgl,
        initial_capacity,
    };

    if (old_picking_bg) wgpuBindGroupRelease(old_picking_bg);
    if (old_picking_bgl) wgpuBindGroupLayoutRelease(old_picking_bgl);
    if (old_gizmo_bg) wgpuBindGroupRelease(old_gizmo_bg);
    if (old_gizmo_bgl) wgpuBindGroupLayoutRelease(old_gizmo_bgl);
}

void EditorPass::ensure_picking_capacity(const webgpu::Device& device, uint32_t count) {
    auto& ready = std::get<Ready>(m_state);
    if (count <= ready.picking_capacity) return;
    uint32_t cap = ready.picking_capacity;
    while (cap < count) cap *= 2;
    ready.picking_uniform_buffer = device.create_buffer(
        k_uniform_align * cap,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));
    if (ready.picking_bind_group) wgpuBindGroupRelease(ready.picking_bind_group);
    ready.picking_bind_group =
        create_bind_group(device.handle(), ready.picking_bind_group_layout,
                          ready.picking_uniform_buffer.handle(), sizeof(PickingUniforms));
    ready.picking_capacity = cap;
}

void EditorPass::ensure_gizmo_capacity(const webgpu::Device& device, uint32_t count) {
    auto& ready = std::get<Ready>(m_state);
    if (count <= ready.gizmo_capacity) return;
    uint32_t cap = ready.gizmo_capacity;
    while (cap < count) cap *= 2;
    ready.gizmo_uniform_buffer = device.create_buffer(
        k_uniform_align * cap,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));
    if (ready.gizmo_bind_group) wgpuBindGroupRelease(ready.gizmo_bind_group);
    ready.gizmo_bind_group =
        create_bind_group(device.handle(), ready.gizmo_bind_group_layout,
                          ready.gizmo_uniform_buffer.handle(), sizeof(GizmoUniforms));
    ready.gizmo_capacity = cap;
}

void EditorPass::add_to_frame_graph(rendering::FrameGraph& fg, const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    auto objects = ctx.world.get_objects();
    auto lights = ctx.world.get_lights();
    auto object_count = static_cast<uint32_t>(objects.size());
    auto light_count = static_cast<uint32_t>(lights.size());

    // Collect active lights that have volume (Distant/Dome are unpickable)
    std::vector<uint32_t> gizmo_light_indices;
    for (uint32_t i = 0; i < light_count; ++i) {
        if (!lights[i].active()) continue;
        if (lights[i]->type == rendering::LightData::Type::Distant ||
            lights[i]->type == rendering::LightData::Type::Dome)
            continue;
        gizmo_light_indices.push_back(i);
    }
    auto gizmo_count = static_cast<uint32_t>(gizmo_light_indices.size());

    // Build picking table: flat mapping from picking_id → prim_path
    m_picking_table.clear();
    m_picking_table.reserve(object_count + gizmo_count);
    for (uint32_t i = 0; i < object_count; ++i) {
        m_picking_table.push_back(objects[i].get_prim_path());
    }
    for (uint32_t slot = 0; slot < gizmo_count; ++slot) {
        m_picking_table.push_back(lights[gizmo_light_indices[slot]].get_prim_path());
    }

    uint32_t total_picking_slots = object_count + gizmo_count;
    if (total_picking_slots > 0) ensure_picking_capacity(ctx.device, total_picking_slots);
    if (gizmo_count > 0) ensure_gizmo_capacity(ctx.device, gizmo_count);

    // ── Create/cache gizmo meshes and collect handles ──────────────────
    struct GizmoDrawInfo {
        WGPUBuffer vertex_buffer;  // lines for color overlay
        uint32_t vertex_count;
        WGPUBuffer pick_vertex_buffer;  // triangles for picking
        uint32_t pick_vertex_count;
    };
    std::vector<GizmoDrawInfo> gizmo_draws;
    gizmo_draws.reserve(gizmo_count);

    auto make_vbuf = [&](const std::vector<glm::vec3>& verts) {
        auto buf = ctx.device.create_buffer(
            verts.size() * sizeof(glm::vec3),
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst));
        wgpuQueueWriteBuffer(ctx.queue, buf.handle(), 0, verts.data(),
                             verts.size() * sizeof(glm::vec3));
        return buf;
    };

    for (uint32_t slot = 0; slot < gizmo_count; ++slot) {
        uint32_t li = gizmo_light_indices[slot];
        auto& mesh =
            get_or_create_pass_data<GizmoMesh>(rendering::PassDataKind::Light, li, ctx.world, [&] {
                auto line_verts = generate_light_verts(lights[li].data());
                auto pick_verts = generate_light_pick_verts(lights[li].data());
                if (line_verts.empty() && pick_verts.empty()) return GizmoMesh{};
                GizmoMesh m;
                if (!line_verts.empty()) {
                    m.vertex_buffer = make_vbuf(line_verts);
                    m.vertex_count = static_cast<uint32_t>(line_verts.size());
                }
                if (!pick_verts.empty()) {
                    m.pick_vertex_buffer = make_vbuf(pick_verts);
                    m.pick_vertex_count = static_cast<uint32_t>(pick_verts.size());
                }
                return m;
            });
        gizmo_draws.push_back({mesh.vertex_buffer.handle(), mesh.vertex_count,
                               mesh.pick_vertex_buffer.handle(), mesh.pick_vertex_count});
    }

    // ── Texture descriptors ────────────────────────────────────────────
    rendering::TextureDesc picking_desc;
    picking_desc.width = ctx.viewport_width;
    picking_desc.height = ctx.viewport_height;
    picking_desc.format = WGPUTextureFormat_R32Uint;
    picking_desc.usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc);
    picking_desc.clear_color = {static_cast<double>(UINT32_MAX), 0, 0, 0};

    rendering::TextureDesc depth_desc;
    depth_desc.width = ctx.viewport_width;
    depth_desc.height = ctx.viewport_height;
    depth_desc.format = WGPUTextureFormat_Depth24Plus;

    auto picking_ids = fg.find_or_create("picking_ids", picking_desc);
    auto picking_depth = fg.find_or_create("picking_depth", depth_desc);

    // ── Upload picking uniforms ────────────────────────────────────────
    auto queue = ctx.queue;
    auto vp = ctx.proj_matrix * ctx.view_matrix;
    auto picking_buf = ready.picking_uniform_buffer.handle();

    for (uint32_t i = 0; i < object_count; ++i) {
        if (!objects[i].active()) continue;
        PickingUniforms u{};
        u.mvp = vp * objects[i]->transform;
        u.object_id = i;
        wgpuQueueWriteBuffer(queue, picking_buf, i * k_uniform_align, &u, sizeof(u));
    }

    auto gizmo_buf = ready.gizmo_uniform_buffer.handle();
    auto camera_pos = ctx.camera_position;
    constexpr float k_min_screen_radius = 0.05f;  // ~5% of viewport height

    for (uint32_t slot = 0; slot < gizmo_count; ++slot) {
        uint32_t li = gizmo_light_indices[slot];
        uint32_t picking_slot = object_count + slot;

        // Scale gizmo to maintain minimum screen-space size
        glm::vec3 light_pos = glm::vec3(lights[li]->transform[3]);
        float dist = glm::length(light_pos - camera_pos);
        float light_radius = (lights[li]->type == rendering::LightData::Type::Rect)
                                 ? std::max(lights[li]->width, lights[li]->height) * 0.5f
                                 : lights[li]->radius;
        float scale = gizmo_distance_scale(dist, light_radius, k_min_screen_radius);
        auto scaled_transform =
            lights[li]->transform * glm::scale(glm::mat4(1.0f), glm::vec3(scale));

        PickingUniforms pu{};
        pu.mvp = vp * scaled_transform;
        pu.object_id = picking_slot;
        wgpuQueueWriteBuffer(queue, picking_buf, picking_slot * k_uniform_align, &pu, sizeof(pu));

        bool is_selected = (ctx.selected_picking_id == picking_slot);
        GizmoUniforms gu{};
        gu.mvp = vp * scaled_transform;
        gu.color =
            is_selected ? glm::vec4(1.0f, 0.8f, 0.2f, 1.0f) : glm::vec4(1.0f, 1.0f, 1.0f, 0.6f);
        wgpuQueueWriteBuffer(queue, gizmo_buf, slot * k_uniform_align, &gu, sizeof(gu));
    }

    // ── Pass 1: Picking ────────────────────────────────────────────────
    auto mesh_picking_pl = ready.picking_pipeline.handle();
    auto line_picking_pl = ready.gizmo_picking_pipeline.handle();
    auto picking_bg = ready.picking_bind_group;
    const auto& world = ctx.world;
    auto obj_count_cap = object_count;

    fg.add_pass("editor_picking")
        .color(picking_ids)
        .depth(picking_depth)
        .execute([=, gizmo_draws = gizmo_draws, &world](WGPURenderPassEncoder pass) {
            auto objs = world.get_objects();
            auto meshes = world.get_meshes();

            // Mesh objects
            wgpuRenderPassEncoderSetPipeline(pass, mesh_picking_pl);
            for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                if (!objs[i].active()) continue;
                uint32_t dyn_offset = i * EditorPass::k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, picking_bg, 1, &dyn_offset);
                const auto& mesh = meshes[objs[i]->mesh_index];
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh->position_buffer.handle(), 0,
                                                     mesh->position_buffer.size());
                wgpuRenderPassEncoderSetIndexBuffer(pass, mesh->index_buffer.handle(),
                                                    WGPUIndexFormat_Uint32, 0,
                                                    mesh->index_buffer.size());
                wgpuRenderPassEncoderDrawIndexed(pass, mesh->index_count, 1, 0, 0, 0);
            }

            // Light gizmo shapes (filled triangles for easier picking)
            wgpuRenderPassEncoderSetPipeline(pass, line_picking_pl);
            for (uint32_t slot = 0; slot < static_cast<uint32_t>(gizmo_draws.size()); ++slot) {
                auto& draw = gizmo_draws[slot];
                if (draw.pick_vertex_count == 0) continue;
                uint32_t picking_slot = obj_count_cap + slot;
                uint32_t dyn_offset = picking_slot * EditorPass::k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, picking_bg, 1, &dyn_offset);
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, draw.pick_vertex_buffer, 0,
                                                     draw.pick_vertex_count * sizeof(glm::vec3));
                wgpuRenderPassEncoderDraw(pass, draw.pick_vertex_count, 1, 0, 0);
            }
        });

    // ── Pass 2: Gizmo color overlay (own transparent texture, composited by editor) ──
    rendering::TextureDesc gizmo_desc;
    gizmo_desc.width = ctx.viewport_width;
    gizmo_desc.height = ctx.viewport_height;
    gizmo_desc.format = WGPUTextureFormat_RGBA8Unorm;
    gizmo_desc.clear_color = {0, 0, 0, 0};

    auto gizmo_overlay = fg.find_or_create("editor_gizmo_overlay", gizmo_desc);

    auto gizmo_color_pl = ready.gizmo_color_pipeline.handle();
    auto gizmo_bg = ready.gizmo_bind_group;

    fg.add_pass("editor_gizmos")
        .color(gizmo_overlay)
        .execute([=, gizmo_draws = gizmo_draws](WGPURenderPassEncoder pass) {
            wgpuRenderPassEncoderSetPipeline(pass, gizmo_color_pl);
            for (uint32_t slot = 0; slot < static_cast<uint32_t>(gizmo_draws.size()); ++slot) {
                auto& draw = gizmo_draws[slot];
                if (draw.vertex_count == 0) continue;
                uint32_t dyn_offset = slot * EditorPass::k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, gizmo_bg, 1, &dyn_offset);
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, draw.vertex_buffer, 0,
                                                     draw.vertex_count * sizeof(glm::vec3));
                wgpuRenderPassEncoderDraw(pass, draw.vertex_count, 1, 0, 0);
            }
        });
}

auto EditorPass::resolve_picking_id(uint32_t id) const noexcept -> const pxr::SdfPath& {
    if (id < static_cast<uint32_t>(m_picking_table.size())) {
        return m_picking_table[id];
    }
    static const pxr::SdfPath k_empty;
    return k_empty;
}

auto EditorPass::find_picking_id(const pxr::SdfPath& prim_path) const noexcept -> uint32_t {
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_picking_table.size()); ++i) {
        if (m_picking_table[i] == prim_path) return i;
    }
    return UINT32_MAX;
}
