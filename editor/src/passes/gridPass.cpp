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
        if (ready->bind_group) {
            wgpuBindGroupRelease(ready->bind_group);
        }
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
    // Capture old state for deferred release (after new state is built)
    WGPUBindGroup old_bind_group = nullptr;
    WGPUBindGroupLayout old_layout = nullptr;
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        old_bind_group = ready->bind_group;
        old_layout = ready->bind_group_layout;
        ready->bind_group = nullptr;
        ready->bind_group_layout = nullptr;
    }

    auto shader_src = get_shader_loader().load("editor/generated/shaders/grid.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    auto uniform_buffer = device.create_buffer(
        sizeof(GridUniforms),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    auto bind_group_layout = editor_grid_shader::create_bind_group_layout_0(device.handle());

    WGPUBindGroupEntry bg_entry = WGPU_BIND_GROUP_ENTRY_INIT;
    bg_entry.binding = 0;
    bg_entry.buffer = uniform_buffer.handle();
    bg_entry.offset = 0;
    bg_entry.size = sizeof(GridUniforms);

    WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg_desc.layout = bind_group_layout;
    bg_desc.entryCount = 1;
    bg_desc.entries = &bg_entry;
    auto bind_group = wgpuDeviceCreateBindGroup(device.handle(), &bg_desc);

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
                        .depth_format(WGPUTextureFormat_Depth24Plus)
                        .depth_write(false)
                        .depth_compare(WGPUCompareFunction_Less)
                        .cull_mode(WGPUCullMode_None)
                        .blend_state(blend_state)
                        .pipeline_layout(pipeline_layout)
                        .build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    m_state = Ready{
        std::move(shader), std::move(pipeline), std::move(uniform_buffer),
        bind_group,        bind_group_layout,
    };

    // Release old resources after new state is built
    if (old_bind_group) wgpuBindGroupRelease(old_bind_group);
    if (old_layout) wgpuBindGroupLayoutRelease(old_layout);
}

void GridPass::add_to_frame_graph(rendering::FrameGraph& fg, const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    rendering::TextureDesc color_desc;
    color_desc.width = ctx.viewport_width;
    color_desc.height = ctx.viewport_height;
    color_desc.format = WGPUTextureFormat_RGBA16Float;
    color_desc.clear_color = {0.15, 0.15, 0.18, 1.0};

    rendering::TextureDesc depth_desc;
    depth_desc.width = ctx.viewport_width;
    depth_desc.height = ctx.viewport_height;
    depth_desc.format = WGPUTextureFormat_Depth24Plus;

    auto color = fg.find_or_create("scene_color", color_desc);
    auto depth = fg.find_or_create("scene_depth", depth_desc);

    auto queue = ctx.queue;
    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    auto cam_pos = ctx.camera_position;
    auto near_plane = ctx.camera.near_plane();
    auto far_plane = ctx.camera.far_plane();
    auto meters_per_unit = ctx.meters_per_unit;
    auto up_axis = ctx.up_axis;
    auto* pipeline_handle = ready.pipeline.handle();
    auto uniform_buf = ready.uniform_buffer.handle();
    auto bind_group = ready.bind_group;

    auto vp_mat = proj_mat * view_mat;
    auto inv_vp_mat = glm::inverse(vp_mat);

    fg.add_pass("grid").color(color).depth_readonly(depth).execute([=](WGPURenderPassEncoder pass) {
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
