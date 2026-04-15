#include "pathTracerPass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/bvh.h>
#include <core/rendering/camera.h>
#include <core/rendering/fallbackPool.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/rendererRegistry.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/webgpu/device.h>
#include <imgui.h>
#include <renderers/pathtracer/generated/pathtracer_shader_metadata.h>
#include <renderers/pathtracer/generated/pt_blit_shader_metadata.h>

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

auto PathTracerPass::name() const noexcept -> std::string_view {
    return "pathtracer";
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
    ++m_output_buffer_version;
    m_pixel_width = width;
    m_pixel_height = height;
    m_frame_count = 0;
}

PathTracerPass::HdrOutputs PathTracerPass::do_add_to_frame_graph(
    rendering::FrameGraph& fg, const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;

    if (!m_uniform_buffer.is_valid()) {
        m_uniform_buffer = ctx.device.create_buffer(
            sizeof(PTUniforms),
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));
    }

    auto compute_desc_layout = fg.bind_group_layout(
        "pathtracer/compute", pathtracer_shader::create_bind_group_layout_0(ctx.device.handle()));

    auto ibl_desc_layout = fg.bind_group_layout(
        "pathtracer/ibl", pathtracer_shader::create_bind_group_layout_1(ctx.device.handle()));

    auto* cp = fg.compute_pipeline("pathtracer_compute")
                   .shader("editor/generated/shaders/pathtracer.wgsl")
                   .entry_point("cs_main")
                   .bind_group_layouts({compute_desc_layout, ibl_desc_layout})
                   .build();

    auto blit_desc_layout = fg.bind_group_layout(
        "pathtracer/blit", pt_blit_shader::create_bind_group_layout_0(ctx.device.handle()));

    auto* bp = fg.render_pipeline("pathtracer_blit")
                   .shader("editor/generated/shaders/pt_blit.wgsl")
                   .color_format(WGPUTextureFormat_RGBA16Float)
                   .cull_mode(WGPUCullMode_None)
                   .bind_group_layouts({blit_desc_layout})
                   .build();

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
    for (const auto& entry : ctx.world.get_lights().span_raw()) {
        if (!entry.active) continue;
        if (entry.value.type == rendering::LightData::Type::Dome) {
            if (!entry.value.env_texture_path.empty()) {
                dome_mod = entry.value.color * entry.value.intensity;
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
    wgpuQueueWriteBuffer(ctx.queue, m_uniform_buffer.handle(), 0, &uniforms, sizeof(uniforms));

    // Capture handles for lambdas
    auto& mat_buf = ctx.world.material_buffer();
    auto& light_buf = ctx.world.light_buffer();
    auto& tri_buf = ctx.world.triangle_buffer();
    auto& inst_buf = ctx.world.instance_buffer();
    auto& bvh_buf = ctx.world.bvh_node_buffer();
    auto width = ctx.viewport_width;
    auto height = ctx.viewport_height;
    auto inst_count = ctx.world.instance_count();

    // --- Create compute descriptor ---
    auto scene_tex_view = ctx.world.texture_array_view();
    auto scene_tex_sampler = ctx.world.texture_sampler();

    auto compute_desc_decl =
        descriptor(fg, compute_desc_layout, "compute_desc")
            .external_buffer(0, m_uniform_buffer.handle(), 0, sizeof(PTUniforms))
            .external_buffer(1, tri_buf.handle(), 0, WGPU_WHOLE_SIZE)
            .external_buffer(2, mat_buf.handle(), 0, WGPU_WHOLE_SIZE)
            .external_buffer(3, light_buf.handle(), 0, WGPU_WHOLE_SIZE)
            .external_buffer(4, m_accum_buffer.handle(), 0, WGPU_WHOLE_SIZE)
            .external_buffer(5, m_output_buffer.handle(), 0, WGPU_WHOLE_SIZE)
            .external_buffer(6, bvh_buf.handle(), 0, WGPU_WHOLE_SIZE)
            .external_view(7, scene_tex_view)
            .sampler(8, scene_tex_sampler)
            .external_buffer(9, inst_buf.handle(), 0, WGPU_WHOLE_SIZE)
            .build();

    // IBL descriptor (slot 1): env cubemap + sampler
    auto& ibl = ctx.world.ibl_resources();
    bool ibl_ready = ibl.is_ready();
    WGPUTextureView ibl_view = ibl_ready ? ibl.env_cubemap_view()
                                         : fg.fallback_pool().view(WGPUTextureFormat_RGBA16Float,
                                                                   WGPUTextureViewDimension_Cube);
    auto ibl_desc_decl =
        descriptor(fg, ibl_desc_layout, "ibl_desc")
            .external_view(0, ibl_view)
            .sampler(1, fg.sampler(WGPUSamplerBindingType_Filtering, WGPUAddressMode_ClampToEdge,
                                   WGPUMipmapFilterMode_Linear))
            .build();

    fg.add_pass("pathtracer_compute")
        .execute([=](rendering::ExecuteContext& exec, WGPUComputePassEncoder enc) {
            if (inst_count == 0 || !ibl_ready) return;
            auto compute_desc = exec.get(compute_desc_decl).bind_group;
            auto ibl_desc = exec.get(ibl_desc_decl).bind_group;
            wgpuComputePassEncoderSetPipeline(enc, cp);
            wgpuComputePassEncoderSetBindGroup(enc, 0, compute_desc, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(enc, 1, ibl_desc, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(enc, (width + 7) / 8, (height + 7) / 8, 1);
        });

    // --- Blit pass ---
    rendering::TextureDesc color_desc;
    color_desc.width = ctx.viewport_width;
    color_desc.height = ctx.viewport_height;
    color_desc.format = WGPUTextureFormat_RGBA16Float;
    color_desc.clear_color = {0.15, 0.15, 0.18, 1.0};
    auto color_decl = create_texture(fg, color_desc, "color");

    // Import the pass-owned output buffer so the FG can track pointer changes
    auto output_buf_decl = import_buffer(fg, m_output_buffer.handle(), m_output_buffer.size(),
                                         m_output_buffer_version, "output");

    // Register blit uniform buffer with frame graph
    rendering::BufferDesc blit_buf_desc{};
    blit_buf_desc.size = sizeof(BlitUniforms);
    blit_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto blit_uniform_buf_decl = create_buffer(fg, blit_buf_desc, "blit_uniforms");

    // Register blit descriptor
    auto blit_desc_decl = descriptor(fg, blit_desc_layout, "blit_desc")
                              .buffer(0, blit_uniform_buf_decl, 0, sizeof(BlitUniforms))
                              .buffer(1, output_buf_decl)
                              .build();

    auto queue = ctx.queue;
    fg.add_pass("pathtracer_blit")
        .color(color_decl)
        .execute([=](rendering::ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto blit_uniform_buf = exec.get(blit_uniform_buf_decl).buffer;
            auto blit_desc = exec.get(blit_desc_decl).bind_group;

            BlitUniforms bu{};
            bu.width = width;
            bu.height = height;
            wgpuQueueWriteBuffer(queue, blit_uniform_buf, 0, &bu, sizeof(bu));

            wgpuRenderPassEncoderSetPipeline(pass, bp);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, blit_desc, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

    return {color_decl, {}, {}};
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
