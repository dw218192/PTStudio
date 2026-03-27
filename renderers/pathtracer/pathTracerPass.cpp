#include "pathTracerPass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/bvh.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/rendererRegistry.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <imgui.h>

#include <glm/glm.hpp>

using namespace pts;
using namespace pts::editor;

REGISTER_RENDERER("Path Trace", PathTracerPass, false);

struct PTUniforms {
    glm::vec3 camera_pos;
    uint32_t frame_index;
    glm::mat4 inv_vp;
    uint32_t width;
    uint32_t height;
    uint32_t instance_count;
    uint32_t light_count;
    uint32_t total_frames;
    uint32_t tlas_node_count;
    uint32_t _pad[2];
};
static_assert(sizeof(PTUniforms) == 112, "PTUniforms must match shader layout");

struct BlitUniforms {
    uint32_t width;
    uint32_t height;
    uint32_t _pad[2];
};
static_assert(sizeof(BlitUniforms) == 16);

static constexpr std::size_t k_min_pixel_buffer_size = 16;

PathTracerPass::~PathTracerPass() {
    if (auto* r = std::get_if<Ready>(&m_state)) {
        if (r->compute_bgl) wgpuBindGroupLayoutRelease(r->compute_bgl);
        if (r->blit_bgl) wgpuBindGroupLayoutRelease(r->blit_bgl);
    }
}

auto PathTracerPass::name() const noexcept -> std::string_view {
    return "pathtracer";
}

auto PathTracerPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

void PathTracerPass::do_renderer_setup(const webgpu::Device& device) {
    if (auto* r = std::get_if<Ready>(&m_state)) {
        if (r->compute_bgl) wgpuBindGroupLayoutRelease(r->compute_bgl);
        if (r->blit_bgl) wgpuBindGroupLayoutRelease(r->blit_bgl);
    }

    // --- Compute pipeline ---
    auto compute_src = get_shader_loader().load("editor/generated/shaders/pathtracer.wgsl");
    auto compute_shader = device.create_shader_module_from_source(compute_src);

    auto uniform_buffer = device.create_buffer(
        sizeof(PTUniforms),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    WGPUBindGroupLayoutEntry ce[10] = {};
    for (int i = 0; i < 10; ++i) ce[i] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;

    ce[0].binding = 0;
    ce[0].visibility = WGPUShaderStage_Compute;
    ce[0].buffer.type = WGPUBufferBindingType_Uniform;
    ce[0].buffer.minBindingSize = sizeof(PTUniforms);

    for (int i = 1; i <= 3; ++i) {
        ce[i].binding = i;
        ce[i].visibility = WGPUShaderStage_Compute;
        ce[i].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    }

    ce[4].binding = 4;
    ce[4].visibility = WGPUShaderStage_Compute;
    ce[4].buffer.type = WGPUBufferBindingType_Storage;

    ce[5].binding = 5;
    ce[5].visibility = WGPUShaderStage_Compute;
    ce[5].buffer.type = WGPUBufferBindingType_Storage;

    // binding 6: BVH nodes (read-only storage)
    ce[6].binding = 6;
    ce[6].visibility = WGPUShaderStage_Compute;
    ce[6].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    ce[6].buffer.minBindingSize = 32;  // sizeof(BVHNode)

    // binding 7: scene texture array
    ce[7].binding = 7;
    ce[7].visibility = WGPUShaderStage_Compute;
    ce[7].texture.sampleType = WGPUTextureSampleType_Float;
    ce[7].texture.viewDimension = WGPUTextureViewDimension_2DArray;
    ce[7].texture.multisampled = false;

    // binding 8: scene texture sampler
    ce[8].binding = 8;
    ce[8].visibility = WGPUShaderStage_Compute;
    ce[8].sampler.type = WGPUSamplerBindingType_Filtering;

    // binding 9: instances (read-only storage)
    ce[9].binding = 9;
    ce[9].visibility = WGPUShaderStage_Compute;
    ce[9].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

    WGPUBindGroupLayoutDescriptor cbgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    cbgl_desc.entryCount = 10;
    cbgl_desc.entries = ce;
    auto compute_bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &cbgl_desc);

    WGPUPipelineLayoutDescriptor cpl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    cpl_desc.bindGroupLayoutCount = 1;
    cpl_desc.bindGroupLayouts = &compute_bgl;
    auto cpl = wgpuDeviceCreatePipelineLayout(device.handle(), &cpl_desc);

    auto compute_pipeline = webgpu::ComputePipelineBuilder(device)
                                .shader(compute_shader)
                                .entry_point("cs_main")
                                .pipeline_layout(cpl)
                                .build();
    wgpuPipelineLayoutRelease(cpl);

    // --- Blit pipeline ---
    auto blit_src = get_shader_loader().load("editor/generated/shaders/pt_blit.wgsl");
    auto blit_shader = device.create_shader_module_from_source(blit_src);

    auto blit_uniform_buffer = device.create_buffer(
        sizeof(BlitUniforms),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    WGPUBindGroupLayoutEntry be[2] = {};
    be[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    be[0].binding = 0;
    be[0].visibility = WGPUShaderStage_Fragment;
    be[0].buffer.type = WGPUBufferBindingType_Uniform;
    be[0].buffer.minBindingSize = sizeof(BlitUniforms);

    be[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    be[1].binding = 1;
    be[1].visibility = WGPUShaderStage_Fragment;
    be[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

    WGPUBindGroupLayoutDescriptor bbgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bbgl_desc.entryCount = 2;
    bbgl_desc.entries = be;
    auto blit_bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &bbgl_desc);

    WGPUPipelineLayoutDescriptor bpl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    bpl_desc.bindGroupLayoutCount = 1;
    bpl_desc.bindGroupLayouts = &blit_bgl;
    auto bpl = wgpuDeviceCreatePipelineLayout(device.handle(), &bpl_desc);

    auto blit_pipeline = webgpu::RenderPipelineBuilder(device)
                             .shader(blit_shader)
                             .color_format(WGPUTextureFormat_RGBA16Float)
                             .cull_mode(WGPUCullMode_None)
                             .pipeline_layout(bpl)
                             .build();
    wgpuPipelineLayoutRelease(bpl);

    m_state = Ready{
        std::move(compute_shader),      std::move(compute_pipeline),
        std::move(uniform_buffer),      compute_bgl,
        std::move(blit_shader),         std::move(blit_pipeline),
        std::move(blit_uniform_buffer), blit_bgl,
    };
}

void PathTracerPass::ensure_pixel_buffers(const webgpu::Device& device, uint32_t width,
                                          uint32_t height) {
    if (m_pixel_width == width && m_pixel_height == height && m_accum_buffer.is_valid()) return;
    auto n = static_cast<std::size_t>(width) * height;
    auto sz = std::max(k_min_pixel_buffer_size, n * 16);
    m_accum_buffer = device.create_buffer(
        sz, static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
    m_output_buffer = device.create_buffer(
        sz, static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
    m_pixel_width = width;
    m_pixel_height = height;
    m_frame_count = 0;
}

void PathTracerPass::do_add_to_frame_graph(rendering::FrameGraph& fg,
                                           const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    auto& r = std::get<Ready>(m_state);

    ensure_pixel_buffers(ctx.device, ctx.viewport_width, ctx.viewport_height);

    auto current_vp = ctx.proj_matrix * ctx.view_matrix;
    if (current_vp != m_prev_vp) {
        m_frame_count = 0;
        m_prev_vp = current_vp;
    }

    // Reset accumulation when the scene changes (instance buffer rebuilt)
    auto current_instance_handle = ctx.world.instance_buffer().handle();
    if (current_instance_handle != m_prev_instance_handle) {
        m_frame_count = 0;
        m_prev_instance_handle = current_instance_handle;
    }

    m_frame_count++;

    PTUniforms uniforms{};
    uniforms.camera_pos = ctx.camera_position;
    uniforms.frame_index = m_frame_count;
    uniforms.inv_vp = glm::inverse(current_vp);
    uniforms.width = ctx.viewport_width;
    uniforms.height = ctx.viewport_height;
    uniforms.instance_count = ctx.world.instance_count();
    uniforms.light_count = ctx.world.gpu_light_count();
    uniforms.total_frames = m_frame_count;
    uniforms.tlas_node_count = ctx.world.tlas_node_count();
    wgpuQueueWriteBuffer(ctx.queue, r.uniform_buffer.handle(), 0, &uniforms, sizeof(uniforms));

    BlitUniforms bu{};
    bu.width = ctx.viewport_width;
    bu.height = ctx.viewport_height;
    wgpuQueueWriteBuffer(ctx.queue, r.blit_uniform_buffer.handle(), 0, &bu, sizeof(bu));

    // Capture handles for lambdas
    auto& mat_buf = ctx.world.material_buffer();
    auto& light_buf = ctx.world.light_buffer();
    auto& tri_buf = ctx.world.triangle_buffer();
    auto& inst_buf = ctx.world.instance_buffer();
    auto& bvh_buf = ctx.world.bvh_node_buffer();
    auto dev = ctx.device.handle();
    auto width = ctx.viewport_width;
    auto height = ctx.viewport_height;
    auto inst_count = ctx.world.instance_count();

    // --- Create compute bind group ---
    auto scene_tex_view = ctx.world.texture_array_view();
    auto scene_tex_sampler = ctx.world.texture_sampler();

    WGPUBindGroupEntry cbe[10] = {};
    for (int i = 0; i < 10; ++i) cbe[i] = WGPU_BIND_GROUP_ENTRY_INIT;
    cbe[0].binding = 0;
    cbe[0].buffer = r.uniform_buffer.handle();
    cbe[0].size = sizeof(PTUniforms);
    cbe[1].binding = 1;
    cbe[1].buffer = tri_buf.handle();
    cbe[1].size = tri_buf.size();
    cbe[2].binding = 2;
    cbe[2].buffer = mat_buf.handle();
    cbe[2].size = mat_buf.size();
    cbe[3].binding = 3;
    cbe[3].buffer = light_buf.handle();
    cbe[3].size = light_buf.size();
    cbe[4].binding = 4;
    cbe[4].buffer = m_accum_buffer.handle();
    cbe[4].size = m_accum_buffer.size();
    cbe[5].binding = 5;
    cbe[5].buffer = m_output_buffer.handle();
    cbe[5].size = m_output_buffer.size();
    cbe[6].binding = 6;
    cbe[6].buffer = bvh_buf.handle();
    cbe[6].size = bvh_buf.size();
    cbe[7].binding = 7;
    cbe[7].textureView = scene_tex_view;
    cbe[8].binding = 8;
    cbe[8].sampler = scene_tex_sampler;
    cbe[9].binding = 9;
    cbe[9].buffer = inst_buf.handle();
    cbe[9].size = inst_buf.size();

    WGPUBindGroupDescriptor cbg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    cbg_desc.layout = r.compute_bgl;
    cbg_desc.entryCount = 10;
    cbg_desc.entries = cbe;
    auto compute_bg = wgpuDeviceCreateBindGroup(dev, &cbg_desc);

    auto* cp = r.compute_pipeline.handle();
    fg.add_pass("pathtracer_compute").execute([=](WGPUComputePassEncoder enc) {
        if (inst_count == 0) {
            wgpuBindGroupRelease(compute_bg);
            return;
        }
        wgpuComputePassEncoderSetPipeline(enc, cp);
        wgpuComputePassEncoderSetBindGroup(enc, 0, compute_bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(enc, (width + 7) / 8, (height + 7) / 8, 1);
        wgpuBindGroupRelease(compute_bg);
    });

    // --- Blit pass ---
    rendering::TextureDesc color_desc;
    color_desc.width = ctx.viewport_width;
    color_desc.height = ctx.viewport_height;
    color_desc.format = WGPUTextureFormat_RGBA16Float;
    color_desc.clear_color = {0.15, 0.15, 0.18, 1.0};
    auto color = fg.find_or_create("scene_color", color_desc);

    WGPUBindGroupEntry bbe[2] = {};
    bbe[0] = WGPU_BIND_GROUP_ENTRY_INIT;
    bbe[0].binding = 0;
    bbe[0].buffer = r.blit_uniform_buffer.handle();
    bbe[0].size = sizeof(BlitUniforms);
    bbe[1] = WGPU_BIND_GROUP_ENTRY_INIT;
    bbe[1].binding = 1;
    bbe[1].buffer = m_output_buffer.handle();
    bbe[1].size = m_output_buffer.size();

    WGPUBindGroupDescriptor bbg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bbg_desc.layout = r.blit_bgl;
    bbg_desc.entryCount = 2;
    bbg_desc.entries = bbe;
    auto blit_bg = wgpuDeviceCreateBindGroup(dev, &bbg_desc);

    auto* bp = r.blit_pipeline.handle();
    fg.add_pass("pathtracer_blit").color(color).execute([=](WGPURenderPassEncoder pass) {
        wgpuRenderPassEncoderSetPipeline(pass, bp);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, blit_bg, 0, nullptr);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        wgpuBindGroupRelease(blit_bg);
    });
}

void PathTracerPass::draw_viewport_controls() {
    ImGui::SameLine();
    ImGui::Text("SPP: %u", m_frame_count);
}

void PathTracerPass::do_draw_imgui() {
}

void PathTracerPass::draw_viewport_overlay(const ViewportOverlayParams& params) {
    IRenderer::draw_viewport_overlay(params);  // forward to children
}
