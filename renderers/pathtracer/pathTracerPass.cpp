#include "pathTracerPass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/bvh.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/outputLayout.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/rendererRegistry.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <imgui.h>

#include <glm/glm.hpp>

using namespace pts;
using namespace pts::editor;
using namespace pts::rendering;

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
    glm::vec3 dome_modulation;
    uint32_t _pad2;
};
static_assert(sizeof(PTUniforms) == 128, "PTUniforms must match shader layout");

struct BlitUniforms {
    uint32_t width;
    uint32_t height;
    uint32_t _pad[2];
};
static_assert(sizeof(BlitUniforms) == 16);

static constexpr std::size_t k_min_pixel_buffer_size = 16;

PathTracerPass::~PathTracerPass() {
    if (auto* r = std::get_if<Ready>(&m_state)) {
        if (r->compute_desc_layout) wgpuBindGroupLayoutRelease(r->compute_desc_layout);
        if (r->ibl_desc_layout) wgpuBindGroupLayoutRelease(r->ibl_desc_layout);
        if (r->blit_desc_layout) wgpuBindGroupLayoutRelease(r->blit_desc_layout);
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
        if (r->compute_desc_layout) wgpuBindGroupLayoutRelease(r->compute_desc_layout);
        if (r->ibl_desc_layout) wgpuBindGroupLayoutRelease(r->ibl_desc_layout);
        if (r->blit_desc_layout) wgpuBindGroupLayoutRelease(r->blit_desc_layout);
    }

    // --- Compute pipeline ---
    auto compute_src = get_shader_loader().load("editor/generated/shaders/pathtracer.wgsl");
    auto compute_shader = device.create_shader_module_from_source(compute_src);

    auto uniform_buffer = device.create_buffer(
        sizeof(PTUniforms),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    auto compute_internal = create_output_layout(
        device,
        {
            OutputSlot::uniform(sizeof(PTUniforms)).visibility(WGPUShaderStage_Compute),
            OutputSlot::storage(0).visibility(WGPUShaderStage_Compute),
            OutputSlot::storage(0).visibility(WGPUShaderStage_Compute),
            OutputSlot::storage(0).visibility(WGPUShaderStage_Compute),
            OutputSlot::storage(0).read_write().visibility(WGPUShaderStage_Compute),
            OutputSlot::storage(0).read_write().visibility(WGPUShaderStage_Compute),
            OutputSlot::storage(32).visibility(WGPUShaderStage_Compute),  // BVH nodes
            OutputSlot::texture(WGPUTextureFormat_RGBA8Unorm, WGPUTextureViewDimension_2DArray)
                .visibility(WGPUShaderStage_Compute),
            OutputSlot::sampler(WGPUSamplerBindingType_Filtering)
                .visibility(WGPUShaderStage_Compute),
            OutputSlot::storage(0).visibility(WGPUShaderStage_Compute),  // instances
        });
    auto compute_desc_layout = compute_internal.layout;
    compute_internal.layout = nullptr;
    compute_internal.release();

    // IBL descriptor layout (group 1): env cubemap + sampler
    auto ibl_internal = create_output_layout(
        device,
        {
            OutputSlot::texture(WGPUTextureFormat_RGBA16Float, WGPUTextureViewDimension_Cube)
                .visibility(WGPUShaderStage_Compute),
            OutputSlot::sampler(WGPUSamplerBindingType_Filtering)
                .visibility(WGPUShaderStage_Compute),
        });
    auto ibl_desc_layout = ibl_internal.layout;
    ibl_internal.layout = nullptr;
    ibl_internal.release();

    WGPUBindGroupLayout compute_desc_layouts[2] = {compute_desc_layout, ibl_desc_layout};
    WGPUPipelineLayoutDescriptor cpl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    cpl_desc.bindGroupLayoutCount = 2;
    cpl_desc.bindGroupLayouts = compute_desc_layouts;
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

    auto blit_internal = create_output_layout(device, {
                                                          OutputSlot::uniform(sizeof(BlitUniforms)),
                                                          OutputSlot::storage(0),
                                                      });
    auto blit_desc_layout = blit_internal.layout;
    blit_internal.layout = nullptr;
    blit_internal.release();

    WGPUPipelineLayoutDescriptor bpl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    bpl_desc.bindGroupLayoutCount = 1;
    bpl_desc.bindGroupLayouts = &blit_desc_layout;
    auto bpl = wgpuDeviceCreatePipelineLayout(device.handle(), &bpl_desc);

    auto blit_pipeline = webgpu::RenderPipelineBuilder(device)
                             .shader(blit_shader)
                             .color_format(WGPUTextureFormat_RGBA16Float)
                             .cull_mode(WGPUCullMode_None)
                             .pipeline_layout(bpl)
                             .build();
    wgpuPipelineLayoutRelease(bpl);

    m_state = Ready{
        std::move(compute_shader), std::move(compute_pipeline),
        std::move(uniform_buffer), compute_desc_layout,
        ibl_desc_layout,           std::move(blit_shader),
        std::move(blit_pipeline),  blit_desc_layout,
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

PathTracerPass::HdrOutputs PathTracerPass::do_add_to_frame_graph(
    rendering::FrameGraph& fg, const rendering::PassContext& ctx) {
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

    // Reset accumulation when lights change (dome color/intensity/HDR)
    auto light_ver = ctx.world.get_light_version();
    if (light_ver != m_prev_light_version) {
        m_frame_count = 0;
        m_prev_light_version = light_ver;
    }

    m_frame_count++;

    // Compute dome modulation: for HDR domes the cubemap has raw HDR values
    // and needs color*intensity applied; for uniform domes the cubemap already
    // has color*intensity baked in, so modulation is (1,1,1).
    glm::vec3 dome_mod{1.0f};
    for (const auto& slot : ctx.world.get_lights()) {
        if (!slot.active()) continue;
        if (slot.data().type == rendering::LightData::Type::Dome) {
            if (!slot.data().env_texture_path.empty()) {
                dome_mod = slot.data().color * slot.data().intensity;
            }
            break;
        }
    }

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
    uniforms.dome_modulation = dome_mod;
    wgpuQueueWriteBuffer(ctx.queue, r.uniform_buffer.handle(), 0, &uniforms, sizeof(uniforms));

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

    // --- Create compute descriptor ---
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
    cbg_desc.layout = r.compute_desc_layout;
    cbg_desc.entryCount = 10;
    cbg_desc.entries = cbe;
    auto compute_bg = wgpuDeviceCreateBindGroup(dev, &cbg_desc);

    // IBL descriptor (group 1): env cubemap + sampler
    auto& ibl = ctx.world.ibl_resources();
    auto& ibl_pipes = ctx.world.ibl_pipelines();
    WGPUBindGroup ibl_bg = nullptr;
    if (ibl.is_ready()) {
        WGPUBindGroupEntry ibe[2] = {};
        ibe[0] = WGPU_BIND_GROUP_ENTRY_INIT;
        ibe[0].binding = 0;
        ibe[0].textureView = ibl.env_cubemap_view();
        ibe[1] = WGPU_BIND_GROUP_ENTRY_INIT;
        ibe[1].binding = 1;
        ibe[1].sampler = ibl_pipes.sampler();

        WGPUBindGroupDescriptor ibg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        ibg_desc.layout = r.ibl_desc_layout;
        ibg_desc.entryCount = 2;
        ibg_desc.entries = ibe;
        ibl_bg = wgpuDeviceCreateBindGroup(dev, &ibg_desc);
    }

    auto* cp = r.compute_pipeline.handle();
    fg.add_pass("pathtracer_compute").execute([=](WGPUComputePassEncoder enc) {
        if (inst_count == 0 || !ibl_bg) {
            wgpuBindGroupRelease(compute_bg);
            if (ibl_bg) wgpuBindGroupRelease(ibl_bg);
            return;
        }
        wgpuComputePassEncoderSetPipeline(enc, cp);
        wgpuComputePassEncoderSetBindGroup(enc, 0, compute_bg, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(enc, 1, ibl_bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(enc, (width + 7) / 8, (height + 7) / 8, 1);
        wgpuBindGroupRelease(compute_bg);
        wgpuBindGroupRelease(ibl_bg);
    });

    // --- Blit pass ---
    rendering::TextureDesc color_desc;
    color_desc.width = ctx.viewport_width;
    color_desc.height = ctx.viewport_height;
    color_desc.format = WGPUTextureFormat_RGBA16Float;
    color_desc.clear_color = {0.15, 0.15, 0.18, 1.0};
    auto color = create_texture(fg, color_desc, "color");

    // Import the pass-owned output buffer so the FG can track pointer changes
    auto output_buf_handle =
        import_buffer(fg, m_output_buffer.handle(), m_output_buffer.size(), "output");

    // Register blit uniform buffer with frame graph
    rendering::BufferDesc blit_buf_desc{};
    blit_buf_desc.size = sizeof(BlitUniforms);
    blit_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto blit_uniform_buf_handle = create_buffer(fg, blit_buf_desc, "blit_uniforms");

    // Register blit descriptor
    auto blit_bg_handle = descriptor(fg, r.blit_desc_layout, "blit_bg")
                              .buffer(0, blit_uniform_buf_handle, 0, sizeof(BlitUniforms))
                              .buffer(1, output_buf_handle)
                              .build();

    auto* bp = r.blit_pipeline.handle();
    auto queue = ctx.queue;
    fg.add_pass("pathtracer_blit").color(color).execute([=, &fg](WGPURenderPassEncoder pass) {
        auto blit_uniform_buf = fg.get_buffer_ref(blit_uniform_buf_handle).handle();
        auto blit_bg = fg.get_descriptor_ref(blit_bg_handle).handle();

        BlitUniforms bu{};
        bu.width = width;
        bu.height = height;
        wgpuQueueWriteBuffer(queue, blit_uniform_buf, 0, &bu, sizeof(bu));

        wgpuRenderPassEncoderSetPipeline(pass, bp);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, blit_bg, 0, nullptr);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    });

    return {color, {}};
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
