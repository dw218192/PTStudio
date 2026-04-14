#include "editorPass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/webgpu/device.h>
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

// -- Uniform structs ----------------------------------------------------

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

// -- EditorPass implementation ------------------------------------------

auto EditorPass::name() const noexcept -> std::string_view {
    return "editor";
}

void EditorPass::render(rendering::FrameGraph& fg, const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;
    ensure_initialized(ctx.device);

    // -- Picking pipeline (mesh objects + light shapes) -----------------
    auto picking_bgl = fg.bind_group_layout(
        "editor/picking", editor_picking_shader::create_bind_group_layout_0(ctx.device.handle()));

    (void) fg.render_pipeline("editor_picking")
        .shader("editor/generated/shaders/picking.wgsl")
        .color_format(WGPUTextureFormat_R32Uint)
        .depth_format(WGPUTextureFormat_Depth32Float)
        .depth_write(true)
        .depth_compare(WGPUCompareFunction_Less)
        .cull_mode(WGPUCullMode_Back)
        .bind_group_layouts({picking_bgl})
        .vertex_layout<editor_picking_shader::VertexLayout>()
        .build();

    // Line-list picking pipeline for wireframe-only lights (e.g. Distant)
    (void) fg.render_pipeline("editor_picking_line")
        .shader("editor/generated/shaders/picking.wgsl")
        .color_format(WGPUTextureFormat_R32Uint)
        .depth_format(WGPUTextureFormat_Depth32Float)
        .depth_write(true)
        .depth_compare(WGPUCompareFunction_Less)
        .cull_mode(WGPUCullMode_None)
        .topology(WGPUPrimitiveTopology_LineList)
        .bind_group_layouts({picking_bgl})
        .vertex_layout<editor_picking_shader::VertexLayout>()
        .build();

    // -- Gizmo color pipeline (wireframe overlay on scene_color) --------
    auto gizmo_bgl = fg.bind_group_layout(
        "editor/gizmo", editor_gizmo_shader::create_bind_group_layout_0(ctx.device.handle()));

    WGPUBlendState blend = {};
    blend.color.operation = WGPUBlendOperation_Add;
    blend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blend.alpha.operation = WGPUBlendOperation_Add;
    blend.alpha.srcFactor = WGPUBlendFactor_One;
    blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;

    (void) fg.render_pipeline("editor_gizmo")
        .shader("editor/generated/shaders/gizmo.wgsl")
        .color_format(WGPUTextureFormat_RGBA8Unorm)
        .blend_state(blend)
        .cull_mode(WGPUCullMode_None)
        .topology(WGPUPrimitiveTopology_LineList)
        .bind_group_layouts({gizmo_bgl})
        .vertex_layout<editor_gizmo_shader::VertexLayout>()
        .build();

    auto objects = ctx.world.get_objects();
    auto lights = ctx.world.get_lights();
    auto object_count = static_cast<uint32_t>(objects.size());
    auto light_count = static_cast<uint32_t>(lights.size());

    // Collect active lights eligible for gizmo rendering (Dome excluded)
    std::vector<uint32_t> gizmo_light_indices;
    for (uint32_t i = 0; i < light_count; ++i) {
        if (!lights[i].active()) continue;
        if (lights[i]->type == rendering::LightData::Type::Dome) continue;
        gizmo_light_indices.push_back(i);
    }
    auto gizmo_count = static_cast<uint32_t>(gizmo_light_indices.size());

    // Build picking table: flat mapping from picking_id -> prim_path
    m_picking_table.clear();
    m_picking_table.reserve(object_count + gizmo_count);
    for (uint32_t i = 0; i < object_count; ++i) {
        m_picking_table.push_back(objects[i].get_prim_path());
    }
    for (uint32_t slot = 0; slot < gizmo_count; ++slot) {
        m_picking_table.push_back(lights[gizmo_light_indices[slot]].get_prim_path());
    }

    uint32_t total_picking_slots = object_count + gizmo_count;

    // Register picking uniform buffer with frame graph
    uint64_t picking_buf_size =
        std::max(uint64_t(1), static_cast<uint64_t>(total_picking_slots)) * k_uniform_align;
    rendering::BufferDesc picking_buf_desc;
    picking_buf_desc.size = picking_buf_size;
    picking_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto picking_buf_decl = create_buffer(fg, picking_buf_desc, "picking_uniforms");

    auto picking_bg_decl = descriptor(fg, picking_bgl, "picking_bg0")
                               .buffer(0, picking_buf_decl, 0, sizeof(PickingUniforms))
                               .build();

    // Register gizmo uniform buffer with frame graph
    uint64_t gizmo_buf_size =
        std::max(uint64_t(1), static_cast<uint64_t>(gizmo_count)) * k_uniform_align;
    rendering::BufferDesc gizmo_buf_desc;
    gizmo_buf_desc.size = gizmo_buf_size;
    gizmo_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto gizmo_buf_decl = create_buffer(fg, gizmo_buf_desc, "gizmo_uniforms");

    auto gizmo_bg_decl = descriptor(fg, gizmo_bgl, "gizmo_bg0")
                             .buffer(0, gizmo_buf_decl, 0, sizeof(GizmoUniforms))
                             .build();

    // -- Create/cache gizmo meshes and collect handles ------------------
    struct GizmoDrawInfo {
        WGPUBuffer vertex_buffer;  // lines for color overlay
        uint32_t vertex_count;
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
                if (line_verts.empty()) return GizmoMesh{};
                GizmoMesh m;
                m.vertex_buffer = make_vbuf(line_verts);
                m.vertex_count = static_cast<uint32_t>(line_verts.size());
                return m;
            });
        gizmo_draws.push_back({mesh.vertex_buffer.handle(), mesh.vertex_count});
    }

    // -- Texture descriptors --------------------------------------------
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
    depth_desc.format = WGPUTextureFormat_Depth32Float;

    auto picking_ids_decl = fg.texture("picking_ids", picking_desc);
    auto picking_depth_decl = fg.texture("picking_depth", depth_desc);

    auto queue = ctx.queue;
    auto vp = ctx.proj_matrix * ctx.view_matrix;
    auto camera_pos = ctx.camera_position;
    auto selected_picking_id = ctx.selected_picking_id;
    constexpr float k_min_screen_radius = 0.05f;

    // -- Pass 1: Picking ------------------------------------------------
    auto mesh_picking_pl = fg.get_render_pipeline("editor_picking");
    auto line_picking_pl = fg.get_render_pipeline("editor_picking_line");
    const auto& world = ctx.world;
    auto obj_count_cap = object_count;
    auto gizmo_light_indices_cap = gizmo_light_indices;

    fg.add_pass("editor_picking")
        .color(picking_ids_decl)
        .depth(picking_depth_decl)
        .execute([=, &world](rendering::ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto objs = world.get_objects();
            auto meshes = world.get_meshes();
            auto picking_buf = exec.get(picking_buf_decl).buffer;
            auto picking_bg = exec.get(picking_bg_decl).bind_group;

            {
                PTS_ZONE_NAMED("picking uniform upload");
                for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                    if (!objs[i].active()) continue;
                    if (!objs[i]->visible) continue;
                    PickingUniforms u{};
                    u.mvp = vp * objs[i]->transform;
                    u.object_id = i;
                    wgpuQueueWriteBuffer(queue, picking_buf, i * k_uniform_align, &u, sizeof(u));
                }
            }

            // Light picking uniforms
            auto lts = world.get_lights();
            for (uint32_t slot = 0; slot < static_cast<uint32_t>(gizmo_light_indices_cap.size());
                 ++slot) {
                uint32_t li = gizmo_light_indices_cap[slot];
                uint32_t picking_slot = obj_count_cap + slot;
                auto transform = lts[li]->transform;
                // Wireframe-only lights need scaled transform to match gizmo visual
                if (lts[li]->mesh_index == UINT32_MAX) {
                    glm::vec3 pos = glm::vec3(transform[3]);
                    float dist = glm::length(pos - camera_pos);
                    float r = (lts[li]->type == rendering::LightData::Type::Distant) ? 0.5f : 0.1f;
                    float scale = gizmo_distance_scale(dist, r, k_min_screen_radius);
                    transform = transform * glm::scale(glm::mat4(1.0f), glm::vec3(scale));
                }
                PickingUniforms pu{};
                pu.mvp = vp * transform;
                pu.object_id = picking_slot;
                wgpuQueueWriteBuffer(queue, picking_buf, picking_slot * k_uniform_align, &pu,
                                     sizeof(pu));
            }

            // Mesh objects
            wgpuRenderPassEncoderSetPipeline(pass, mesh_picking_pl);
            for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                if (!objs[i].active()) continue;
                if (!objs[i]->visible) continue;
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

            // Light proxy meshes (same pipeline as mesh objects)
            for (uint32_t slot = 0; slot < static_cast<uint32_t>(gizmo_light_indices_cap.size());
                 ++slot) {
                uint32_t li = gizmo_light_indices_cap[slot];
                if (lts[li]->mesh_index == UINT32_MAX) continue;
                uint32_t picking_slot = obj_count_cap + slot;
                uint32_t dyn_offset = picking_slot * EditorPass::k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, picking_bg, 1, &dyn_offset);
                const auto& mesh = meshes[lts[li]->mesh_index];
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh->position_buffer.handle(), 0,
                                                     mesh->position_buffer.size());
                wgpuRenderPassEncoderSetIndexBuffer(pass, mesh->index_buffer.handle(),
                                                    WGPUIndexFormat_Uint32, 0,
                                                    mesh->index_buffer.size());
                wgpuRenderPassEncoderDrawIndexed(pass, mesh->index_count, 1, 0, 0, 0);
            }

            // Wireframe-only light picking (e.g. Distant) via line-list pipeline
            wgpuRenderPassEncoderSetPipeline(pass, line_picking_pl);
            for (uint32_t slot = 0; slot < static_cast<uint32_t>(gizmo_light_indices_cap.size());
                 ++slot) {
                uint32_t li = gizmo_light_indices_cap[slot];
                if (lts[li]->mesh_index != UINT32_MAX) continue;
                auto& draw = gizmo_draws[slot];
                if (draw.vertex_count == 0) continue;
                uint32_t picking_slot = obj_count_cap + slot;
                uint32_t dyn_offset = picking_slot * EditorPass::k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, picking_bg, 1, &dyn_offset);
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, draw.vertex_buffer, 0,
                                                     draw.vertex_count * sizeof(glm::vec3));
                wgpuRenderPassEncoderDraw(pass, draw.vertex_count, 1, 0, 0);
            }
        });

    // -- Pass 2: Gizmo color overlay (own transparent texture, composited by editor) --
    rendering::TextureDesc gizmo_desc;
    gizmo_desc.width = ctx.viewport_width;
    gizmo_desc.height = ctx.viewport_height;
    gizmo_desc.format = WGPUTextureFormat_RGBA8Unorm;
    gizmo_desc.clear_color = {0, 0, 0, 0};

    auto gizmo_overlay_decl = fg.texture("editor_gizmo_overlay", gizmo_desc);

    auto gizmo_color_pl = fg.get_render_pipeline("editor_gizmo");

    fg.add_pass("editor_gizmos")
        .color(gizmo_overlay_decl)
        .execute([=, &world, gizmo_draws = std::move(gizmo_draws)](rendering::ExecuteContext& exec,
                                                                   WGPURenderPassEncoder pass) {
            auto gizmo_buf = exec.get(gizmo_buf_decl).buffer;
            auto gizmo_bg = exec.get(gizmo_bg_decl).bind_group;

            // Upload gizmo uniforms
            auto lts = world.get_lights();
            for (uint32_t slot = 0; slot < static_cast<uint32_t>(gizmo_light_indices_cap.size());
                 ++slot) {
                uint32_t li = gizmo_light_indices_cap[slot];
                uint32_t picking_slot = obj_count_cap + slot;
                glm::vec3 light_pos = glm::vec3(lts[li]->transform[3]);
                float dist = glm::length(light_pos - camera_pos);
                float light_radius;
                if (lts[li]->type == rendering::LightData::Type::Rect)
                    light_radius = std::max(lts[li]->width, lts[li]->height) * 0.5f;
                else if (lts[li]->type == rendering::LightData::Type::Distant)
                    light_radius = 0.5f;
                else
                    light_radius = lts[li]->radius;
                float scale = gizmo_distance_scale(dist, light_radius, k_min_screen_radius);
                auto scaled_transform =
                    lts[li]->transform * glm::scale(glm::mat4(1.0f), glm::vec3(scale));
                bool is_selected = (selected_picking_id == picking_slot);
                GizmoUniforms gu{};
                gu.mvp = vp * scaled_transform;
                gu.color = is_selected ? glm::vec4(1.0f, 0.8f, 0.2f, 1.0f)
                                       : glm::vec4(1.0f, 1.0f, 1.0f, 0.6f);
                wgpuQueueWriteBuffer(queue, gizmo_buf, slot * k_uniform_align, &gu, sizeof(gu));
            }

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
