#include "gridPass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/webgpu/device.h>
#include <grid_shader_metadata.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace pts;
using namespace pts::editor;

struct GridUniforms {
    glm::mat4 inv_vp;
    glm::mat4 vp;
    glm::vec3 camera_pos;
    float near_plane;
    float far_plane;
    float meters_per_unit;
    int32_t up_axis;
    float _pad;
};
static_assert(sizeof(GridUniforms) == 160, "GridUniforms must match shader std140 layout");

auto GridPass::name() const noexcept -> std::string_view {
    return "grid";
}

void GridPass::render(rendering::FrameGraph& fg, const rendering::PassContext& ctx,
                      rendering::TextureDeclHandle color, rendering::TextureDeclHandle depth) {
    PTS_ZONE_SCOPED;
    PRECONDITION(color);
    PRECONDITION(depth);
    ensure_initialized(ctx.device);

    auto descriptor_layout = fg.bind_group_layout(
        "grid/desc", editor_grid_shader::create_bind_group_layout_0(ctx.device.handle()));

    // Premultiplied alpha blending
    WGPUBlendState blend_state = {};
    blend_state.color.srcFactor = WGPUBlendFactor_One;
    blend_state.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blend_state.color.operation = WGPUBlendOperation_Add;
    blend_state.alpha.srcFactor = WGPUBlendFactor_One;
    blend_state.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blend_state.alpha.operation = WGPUBlendOperation_Add;

    auto* pipeline_handle = fg.render_pipeline("grid")
                                .shader("editor/generated/shaders/grid.wgsl")
                                .color_format(WGPUTextureFormat_RGBA16Float)
                                .depth_format(WGPUTextureFormat_Depth32Float)
                                .depth_write(false)
                                .depth_compare(WGPUCompareFunction_Less)
                                .cull_mode(WGPUCullMode_None)
                                .blend_state(blend_state)
                                .bind_group_layouts({descriptor_layout})
                                .build();

    // Register uniform buffer with frame graph
    rendering::BufferDesc buf_desc{};
    buf_desc.size = sizeof(GridUniforms);
    buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_decl = create_buffer(fg, buf_desc, "uniforms");

    // Register descriptor with frame graph
    auto bg_decl = descriptor(fg, descriptor_layout, "bg0")
                       .buffer(0, uniform_buf_decl, 0, sizeof(GridUniforms))
                       .build();

    auto queue = ctx.queue;
    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    auto cam_pos = ctx.camera_position;
    auto near_plane = ctx.camera.near_plane();
    auto far_plane = ctx.camera.far_plane();
    auto meters_per_unit = ctx.meters_per_unit;
    auto up_axis = ctx.up_axis;
    auto vp_mat = proj_mat * view_mat;
    auto inv_vp_mat = glm::inverse(vp_mat);

    fg.add_pass("grid").color(color).depth_readonly(depth).execute(
        [=](rendering::ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto uniform_buf = exec.get(uniform_buf_decl).buffer;
            auto desc_group = exec.get(bg_decl).bind_group;
            GridUniforms gu;
            gu.inv_vp = inv_vp_mat;
            gu.vp = vp_mat;
            gu.camera_pos = cam_pos;
            gu.near_plane = near_plane;
            gu.far_plane = far_plane;
            gu.meters_per_unit = meters_per_unit;
            gu.up_axis = static_cast<int32_t>(up_axis);
            gu._pad = 0.0f;
            wgpuQueueWriteBuffer(queue, uniform_buf, 0, &gu, sizeof(gu));
            wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, desc_group, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });
}
