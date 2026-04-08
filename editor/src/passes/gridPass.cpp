#include "gridPass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
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

GridPass::~GridPass() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group_layout) {
            wgpuBindGroupLayoutRelease(ready->bind_group_layout);
        }
    }
}

auto GridPass::name() const noexcept -> std::string_view {
    return "grid";
}

auto GridPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

void GridPass::do_setup(const webgpu::Device& device) {
    WGPUBindGroupLayout old_layout = nullptr;
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        old_layout = ready->bind_group_layout;
        ready->bind_group_layout = nullptr;
    }

    auto shader_src = get_shader_loader().load("editor/generated/shaders/grid.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    auto bind_group_layout = editor_grid_shader::create_bind_group_layout_0(device.handle());

    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &bind_group_layout;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

    // Premultiplied alpha blending
    WGPUBlendState blend_state = {};
    blend_state.color.srcFactor = WGPUBlendFactor_One;
    blend_state.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blend_state.color.operation = WGPUBlendOperation_Add;
    blend_state.alpha.srcFactor = WGPUBlendFactor_One;
    blend_state.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blend_state.alpha.operation = WGPUBlendOperation_Add;

    auto pipeline = webgpu::RenderPipelineBuilder(device)
                        .shader(shader)
                        .color_format(WGPUTextureFormat_RGBA16Float)
                        .depth_format(WGPUTextureFormat_Depth32Float)
                        .depth_write(false)
                        .depth_compare(WGPUCompareFunction_Less)
                        .cull_mode(WGPUCullMode_None)
                        .blend_state(blend_state)
                        .pipeline_layout(pipeline_layout)
                        .build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    m_state = Ready{
        std::move(shader),
        std::move(pipeline),
        bind_group_layout,
    };

    if (old_layout) wgpuBindGroupLayoutRelease(old_layout);
}

void GridPass::render(rendering::FrameGraph& fg, const rendering::PassContext& ctx,
                      rendering::TextureHandle color, rendering::TextureHandle depth) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    PRECONDITION(color.is_valid());
    PRECONDITION(depth.is_valid());
    auto& ready = std::get<Ready>(m_state);

    // Register uniform buffer with frame graph
    rendering::BufferDesc buf_desc{};
    buf_desc.size = sizeof(GridUniforms);
    buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_handle = create_buffer(fg, buf_desc, "uniforms");

    // Register bind group with frame graph
    rendering::BindGroupEntry entry{};
    entry.binding = 0;
    entry.buffer = uniform_buf_handle;
    entry.buffer_size = sizeof(GridUniforms);

    rendering::BindGroupDesc bg_desc{};
    bg_desc.layout = ready.bind_group_layout;
    bg_desc.entries = {entry};
    auto bg_handle = create_bind_group(fg, std::move(bg_desc), "bg0");

    auto queue = ctx.queue;
    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    auto cam_pos = ctx.camera_position;
    auto near_plane = ctx.camera.near_plane();
    auto far_plane = ctx.camera.far_plane();
    auto meters_per_unit = ctx.meters_per_unit;
    auto up_axis = ctx.up_axis;
    auto* pipeline_handle = ready.pipeline.handle();

    auto vp_mat = proj_mat * view_mat;
    auto inv_vp_mat = glm::inverse(vp_mat);

    fg.add_pass("grid").color(color).depth_readonly(depth).execute(
        [=, &fg](WGPURenderPassEncoder pass) {
            auto uniform_buf = fg.get_buffer_ref(uniform_buf_handle).handle();
            auto bind_group = fg.get_bind_group_ref(bg_handle).handle();
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
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });
}
