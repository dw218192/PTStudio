#include "wireframePass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/outputLayout.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/rendererRegistry.h>
#include <core/rendering/webgpu/device.h>
#include <core/rendering/wireframeIndices.h>
#include <wireframe_shader_metadata.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace pts;
using namespace pts::editor;
using namespace pts::rendering;

REGISTER_RENDERER("Wireframe", WireframePass);

struct WireframeMesh {
    webgpu::Buffer index_buffer;
    uint32_t index_count;
};

struct WireframeUniforms {
    glm::mat4 mvp;
};
static_assert(sizeof(WireframeUniforms) == 64, "WireframeUniforms must match shader std140 layout");
static_assert(WireframePass::k_uniform_align >= sizeof(WireframeUniforms),
              "Alignment must be >= uniform struct size");

auto WireframePass::name() const noexcept -> std::string_view {
    return "wireframe";
}

WireframePass::HdrOutputs WireframePass::do_add_to_frame_graph(rendering::FrameGraph& fg,
                                                               const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;

    auto descriptor_layout = fg.bind_group_layout(
        "wireframe/desc", {rendering::OutputSlot::uniform(sizeof(WireframeUniforms))
                               .dynamic()
                               .visibility(static_cast<WGPUShaderStage>(
                                   WGPUShaderStage_Vertex | WGPUShaderStage_Fragment))});

    auto* pipeline_handle = fg.render_pipeline("wireframe")
                                .shader("editor/generated/shaders/wireframe.wgsl")
                                .color_format(WGPUTextureFormat_RGBA16Float)
                                .depth_format(WGPUTextureFormat_Depth32Float)
                                .depth_write(true)
                                .depth_compare(WGPUCompareFunction_Less)
                                .cull_mode(WGPUCullMode_None)
                                .topology(WGPUPrimitiveTopology_LineList)
                                .bind_group_layouts({descriptor_layout})
                                .vertex_layout<editor_wireframe_shader::VertexLayout>()
                                .build();

    auto objects = ctx.world.get_objects();
    auto meshes = ctx.world.get_meshes();
    auto object_count = static_cast<uint32_t>(objects.size());

    // Register per-object uniform buffer with frame graph
    uint64_t needed_size =
        std::max(uint64_t(1), static_cast<uint64_t>(object_count)) * k_uniform_align;
    rendering::BufferDesc buf_desc;
    buf_desc.size = needed_size;
    buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_decl = create_buffer(fg, buf_desc, "uniforms");

    // Register descriptor
    auto bg_decl = descriptor(fg, descriptor_layout, "bg0")
                       .buffer(0, uniform_buf_decl, 0, sizeof(WireframeUniforms))
                       .build();

    rendering::TextureDesc color_desc;
    color_desc.width = ctx.viewport_width;
    color_desc.height = ctx.viewport_height;
    color_desc.format = WGPUTextureFormat_RGBA16Float;
    color_desc.clear_color = {0.15, 0.15, 0.18, 1.0};

    rendering::TextureDesc depth_desc;
    depth_desc.width = ctx.viewport_width;
    depth_desc.height = ctx.viewport_height;
    depth_desc.format = WGPUTextureFormat_Depth32Float;

    auto color_decl = create_texture(fg, color_desc, "color");
    auto depth_decl = create_texture(fg, depth_desc, "depth");

    auto queue = ctx.queue;
    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    const auto& world = ctx.world;

    {
        PTS_ZONE_NAMED("wireframe mesh cache");
        for (uint32_t i = 0; i < object_count; ++i) {
            if (!objects[i].active()) continue;
            if (!objects[i]->visible) continue;
            const auto& obj = objects[i];
            get_or_create_pass_data<WireframeMesh>(
                rendering::PassDataKind::Mesh, obj->mesh_index, ctx.world, [&]() {
                    const auto& mesh = meshes[obj->mesh_index];
                    auto indices = expand_wireframe_indices(mesh->cpu_indices);
                    auto buf = ctx.device.create_buffer(
                        indices.size() * sizeof(uint32_t),
                        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Index |
                                                     WGPUBufferUsage_CopyDst));
                    wgpuQueueWriteBuffer(queue, buf.handle(), 0, indices.data(),
                                         indices.size() * sizeof(uint32_t));
                    return WireframeMesh{std::move(buf), static_cast<uint32_t>(indices.size())};
                });
        }
    }

    fg.add_pass("wireframe")
        .color(color_decl)
        .depth(depth_decl)
        .execute([=, &world](rendering::ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto objs = world.get_objects();
            auto mshs = world.get_meshes();
            auto uniform_buf = exec.get(uniform_buf_decl).buffer;
            auto desc_group = exec.get(bg_decl).bind_group;

            {
                PTS_ZONE_NAMED("wireframe uniform upload");
                for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                    if (!objs[i].active()) continue;
                    if (!objs[i]->visible) continue;
                    WireframeUniforms u{};
                    u.mvp = proj_mat * view_mat * objs[i]->transform;
                    wgpuQueueWriteBuffer(queue, uniform_buf, i * k_uniform_align, &u, sizeof(u));
                }
            }

            wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
            for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                if (!objs[i].active()) continue;
                if (!objs[i]->visible) continue;
                uint32_t dyn_offset = i * k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, desc_group, 1, &dyn_offset);
                const auto& mesh = mshs[objs[i]->mesh_index];
                auto& wf = get_or_create_pass_data<WireframeMesh>(
                    rendering::PassDataKind::Mesh, objs[i]->mesh_index, world, nullptr);
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh->vertex_buffer.handle(), 0,
                                                     mesh->vertex_buffer.size());
                wgpuRenderPassEncoderSetIndexBuffer(pass, wf.index_buffer.handle(),
                                                    WGPUIndexFormat_Uint32, 0,
                                                    wf.index_buffer.size());
                wgpuRenderPassEncoderDrawIndexed(pass, wf.index_count, 1, 0, 0, 0);
            }
        });

    return {color_decl, depth_decl};
}
