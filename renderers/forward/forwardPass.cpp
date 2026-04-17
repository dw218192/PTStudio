#include "forwardPass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/contactShadowPass.h>
#include <core/rendering/fallbackPool.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/gbufferPass.h>
#include <core/rendering/halfFloat.h>
#include <core/rendering/iblResources.h>
#include <core/rendering/ltcData.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/rendererRegistry.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/ssaoPass.h>
#include <core/rendering/webgpu/device.h>
#include <renderers/forward/generated/shader_metadata.h>
#include <renderers/forward/generated/skybox_shader_metadata.h>

#include <glm/glm.hpp>
#include <vector>

using namespace pts;
using namespace pts::editor;
using namespace pts::rendering;

REGISTER_RENDERER("Forward", ForwardPass);

ForwardPass::ForwardPass(const rendering::ShaderLoader& sl) : IRenderer(sl) {
    add_pass<rendering::GBufferPass>(sl);
    add_pass<rendering::ShadowMapPass>(sl);
    add_pass<rendering::SSAOPass>(sl);
    add_pass<rendering::ContactShadowPass>(sl);
}

struct ForwardUniforms {
    glm::mat4 mvp;
    glm::mat4 model;
    glm::vec3 camera_pos;
    float time;
    uint32_t material_index;
    uint32_t light_count;
    glm::vec2 viewport_size;
    glm::vec3 ibl_dome_modulation;
    uint32_t ibl_mip_count;
};
static_assert(sizeof(ForwardUniforms) == 176, "ForwardUniforms must match shader std140 layout");
static_assert(ForwardPass::k_uniform_align >= sizeof(ForwardUniforms),
              "Alignment must be >= uniform struct size");

struct SkyboxUniforms {
    glm::mat4 inv_vp;           // 64 bytes
    glm::vec3 camera_pos;       // 12 bytes
    float _pad0;                // 4 bytes
    glm::vec3 dome_modulation;  // 12 bytes
    float _pad1;                // 4 bytes
};
static_assert(sizeof(SkyboxUniforms) == 96, "SkyboxUniforms must match shader std140 layout");

static constexpr rendering::IPass::DebugTarget k_debug_targets[] = {
    {"Direct Diffuse", "debug_Direct Diffuse"},   {"Direct Specular", "debug_Direct Specular"},
    {"IBL Diffuse", "debug_IBL Diffuse"},         {"IBL Specular", "debug_IBL Specular"},
    {"Prefiltered Env", "debug_Prefiltered Env"}, {"BRDF LUT", "debug_BRDF LUT"},
};
static constexpr uint32_t k_debug_target_count =
    static_cast<uint32_t>(sizeof(k_debug_targets) / sizeof(k_debug_targets[0]));

auto ForwardPass::name() const noexcept -> std::string_view {
    return "forward";
}

auto ForwardPass::renderer_debug_targets() const noexcept
    -> std::pair<const DebugTarget*, uint32_t> {
    return {k_debug_targets, k_debug_target_count};
}

static void init_ltc_textures(rendering::FrameGraph& fg, const pts::webgpu::Device& /*device*/) {
    constexpr uint32_t n = static_cast<uint32_t>(rendering::k_ltc_size);

    // Static upload data -- must outlive the first compile() so the decl
    // keeps a valid pointer until wgpuQueueWriteTexture runs.
    static const auto k_ltc_mat_half = [] {
        constexpr uint32_t sz = static_cast<uint32_t>(rendering::k_ltc_size);
        std::vector<uint16_t> v(sz * sz * 4);
        for (size_t i = 0; i < sz * sz * 4; ++i) {
            v[i] = rendering::float_to_half(rendering::k_ltc_mat[i]);
        }
        return v;
    }();
    static const auto k_ltc_amp_half = [] {
        constexpr uint32_t sz = static_cast<uint32_t>(rendering::k_ltc_size);
        std::vector<uint16_t> v(sz * sz * 2);
        for (size_t i = 0; i < sz * sz * 2; ++i) {
            v[i] = rendering::float_to_half(rendering::k_ltc_amp[i]);
        }
        return v;
    }();

    // M^(-1) matrix texture: RGBA16Float
    {
        WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        desc.size = {n, n, 1};
        desc.format = WGPUTextureFormat_RGBA16Float;
        desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding |
                                                   WGPUTextureUsage_CopyDst);
        desc.mipLevelCount = 1;
        desc.sampleCount = 1;
        desc.dimension = WGPUTextureDimension_2D;
        fg.texture(std::string("ltc_mat"), desc, k_ltc_mat_half.data(),
                   static_cast<uint64_t>(k_ltc_mat_half.size() * sizeof(uint16_t)),
                   static_cast<uint32_t>(n * 4 * sizeof(uint16_t)));
    }

    // Amplitude texture: RG16Float
    {
        WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        desc.size = {n, n, 1};
        desc.format = WGPUTextureFormat_RG16Float;
        desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding |
                                                   WGPUTextureUsage_CopyDst);
        desc.mipLevelCount = 1;
        desc.sampleCount = 1;
        desc.dimension = WGPUTextureDimension_2D;
        fg.texture(std::string("ltc_amp"), desc, k_ltc_amp_half.data(),
                   static_cast<uint64_t>(k_ltc_amp_half.size() * sizeof(uint16_t)),
                   static_cast<uint32_t>(n * 2 * sizeof(uint16_t)));
    }
}

ForwardPass::HdrOutputs ForwardPass::do_add_to_frame_graph(rendering::FrameGraph& fg,
                                                           const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;

    // Static textures used by the forward pipeline and its fallback paths.
    {
        PTS_ZONE_NAMED("fwd: ltc+fallback init");
        init_ltc_textures(fg, ctx.device);
        {
            static constexpr uint8_t k_black_cube_pixels[6 * 4] = {};  // 6 * 1x1 RGBA8 pixels
            WGPUTextureDescriptor cube_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
            cube_desc.size = {1, 1, 6};
            cube_desc.format = WGPUTextureFormat_RGBA8Unorm;
            cube_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding |
                                                            WGPUTextureUsage_CopyDst);
            cube_desc.mipLevelCount = 1;
            cube_desc.sampleCount = 1;
            cube_desc.dimension = WGPUTextureDimension_2D;
            fg.texture("forward_ibl_fallback_cube", cube_desc, k_black_cube_pixels,
                       sizeof(k_black_cube_pixels), 4, WGPUTextureViewDimension_Cube);
        }
        {
            static constexpr uint8_t k_black_2d_pixels[4] = {};  // 1x1 RGBA8
            WGPUTextureDescriptor tex_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
            tex_desc.size = {1, 1, 1};
            tex_desc.format = WGPUTextureFormat_RGBA8Unorm;
            tex_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding |
                                                           WGPUTextureUsage_CopyDst);
            tex_desc.mipLevelCount = 1;
            tex_desc.sampleCount = 1;
            tex_desc.dimension = WGPUTextureDimension_2D;
            fg.texture("forward_ibl_fallback_2d", tex_desc, k_black_2d_pixels,
                       sizeof(k_black_2d_pixels), 4);
        }
    }

    // --- Layout setup for the forward pipeline (from shader reflection) ---
    // Register forward's layouts (including consumer layouts) BEFORE pre-passes
    // so the FG cache is keyed to the shader-derived layouts. Pre-passes that
    // later call fg.bind_group_layout with the same name will receive the
    // cached handles (their own supplied layouts are released as duplicates).
    auto descriptor_layout = fg.bind_group_layout(
        "forward/desc", forward_shader::create_bind_group_layout_0(ctx.device.handle()));

    auto ibl_desc_layout = fg.bind_group_layout(
        "forward/ibl", forward_shader::create_bind_group_layout_2(ctx.device.handle()));

    auto skybox_desc_layout = fg.bind_group_layout(
        "forward/skybox", skybox_shader::create_bind_group_layout_0(ctx.device.handle()));

    auto shadow_consumer_bgl = fg.bind_group_layout(
        "shadow_map/consumer", forward_shader::create_bind_group_layout_1(ctx.device.handle()));
    auto cs_consumer_bgl = fg.bind_group_layout(
        "contact_shadow/consumer", forward_shader::create_bind_group_layout_3(ctx.device.handle()));

    // Pre-passes: G-buffer (depth + normals) and shadow maps
    rendering::GBufferPass::Outputs gbuf_out;
    {
        PTS_ZONE_NAMED("fwd: gbuffer add_to_frame_graph");
        if (auto* gbuf = get_pass<rendering::GBufferPass>()) {
            gbuf_out = gbuf->add_to_frame_graph(fg, ctx, {});
        }
    }

    rendering::ShadowMapPass::Outputs shadow_out{};
    {
        PTS_ZONE_NAMED("fwd: shadow add_to_frame_graph");
        if (auto* shadow = get_pass<rendering::ShadowMapPass>()) {
            shadow_out = shadow->add_to_frame_graph(fg, ctx, {});
        }
    }

    auto [dbg_targets_setup, dbg_count_setup] = effective_debug_targets();
    WGPUShaderModule shader;
    {
        PTS_ZONE_NAMED("fwd: load_pass_shader_module");
        shader = load_pass_shader_module(fg, "renderers/forward/generated/shaders/forward.wgsl");
    }

    auto builder = fg.render_pipeline("forward")
                       .shader_module(shader)
                       .color_format(WGPUTextureFormat_RGBA16Float, 0)
                       .depth_format(WGPUTextureFormat_Depth32Float)
                       .depth_write(true)
                       .depth_compare(WGPUCompareFunction_LessEqual)
                       .cull_mode(WGPUCullMode_Back)
                       .bind_group_layouts({descriptor_layout, shadow_consumer_bgl, ibl_desc_layout,
                                            cs_consumer_bgl})
                       .vertex_layout<forward_shader::VertexLayout>();
    for (uint32_t i = 0; i < dbg_count_setup; ++i) {
        builder.color_format(WGPUTextureFormat_RGBA8Unorm, i + 1);
    }
    auto* pipeline_handle = builder.build();

    auto skybox_builder = fg.render_pipeline("forward_skybox")
                              .shader("renderers/forward/generated/shaders/skybox.wgsl")
                              .color_format(WGPUTextureFormat_RGBA16Float, 0)
                              .depth_format(WGPUTextureFormat_Depth32Float)
                              .depth_write(false)
                              .depth_compare(WGPUCompareFunction_LessEqual)
                              .cull_mode(WGPUCullMode_None)
                              .bind_group_layouts({skybox_desc_layout});
    for (uint32_t i = 0; i < dbg_count_setup; ++i) {
        skybox_builder.color_format(WGPUTextureFormat_RGBA8Unorm, i + 1)
            .write_mask(WGPUColorWriteMask_None, i + 1);
    }
    auto* skybox_pipeline_handle = skybox_builder.build();

    auto objs_raw = ctx.world.get_objects().span_raw();
    auto object_count = static_cast<uint32_t>(objs_raw.size());

    // Count proxy lights (lights with active mesh proxies) for uniform buffer sizing
    auto lights_raw = ctx.world.get_lights().span_raw();
    uint32_t proxy_light_count = 0;
    for (uint32_t li = 0; li < static_cast<uint32_t>(lights_raw.size()); ++li) {
        if (!lights_raw[li].active) continue;
        if (lights_raw[li].value.mesh_index == UINT32_MAX) continue;
        ++proxy_light_count;
    }

    uint32_t total_slots = object_count + proxy_light_count;

    // Import external buffers from RenderWorld
    auto& light_buf = ctx.world.light_buffer();
    auto& mat_buf = ctx.world.material_buffer();
    auto light_count = ctx.world.gpu_light_count();
    auto light_buf_decl = import_buffer(fg, light_buf.handle(), light_buf.size(),
                                        ctx.world.light_buffer_version(), "world_lights");
    auto mat_buf_decl = import_buffer(fg, mat_buf.handle(), mat_buf.size(),
                                      ctx.world.material_buffer_version(), "world_materials");

    auto scene_tex_view = ctx.world.texture_array_view();
    auto scene_tex_sampler = ctx.world.texture_sampler();

    // Managed uniform buffer
    uint64_t uniform_needed =
        std::max(uint64_t(1), static_cast<uint64_t>(total_slots)) * k_uniform_align;
    rendering::BufferDesc uniform_buf_desc;
    uniform_buf_desc.size = uniform_needed;
    uniform_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_decl = create_buffer(fg, uniform_buf_desc, "uniforms");

    // Look up LTC textures from persistent cache (first-call init above).
    auto ltc_mat_decl = fg.find_texture("ltc_mat");
    auto ltc_amp_decl = fg.find_texture("ltc_amp");
    INVARIANT(ltc_mat_decl && ltc_amp_decl);
    auto ltc_sampler = fg.sampler(WGPUSamplerBindingType_Filtering);

    // Descriptor 0: materials, lights, uniforms, LTC, scene textures
    auto desc0_decl = descriptor(fg, descriptor_layout, "desc0")
                          .buffer(0, uniform_buf_decl, 0, sizeof(ForwardUniforms))
                          .buffer(1, mat_buf_decl)
                          .buffer(2, light_buf_decl)
                          .texture(3, ltc_mat_decl)
                          .texture(4, ltc_amp_decl)
                          .sampler(5, ltc_sampler)
                          .external_view(6, scene_tex_view)
                          .sampler(7, scene_tex_sampler)
                          .build();

    // Descriptor 1: shadow (child-owned)
    PRECONDITION(shadow_out.consumer_desc);

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
    auto depth_decl = gbuf_out.depth ? gbuf_out.depth : create_texture(fg, depth_desc, "depth");

    auto [eff_debug_targets, eff_debug_count] = effective_debug_targets();

    rendering::TextureDesc debug_desc;
    debug_desc.width = ctx.viewport_width;
    debug_desc.height = ctx.viewport_height;
    debug_desc.format = WGPUTextureFormat_RGBA8Unorm;
    debug_desc.clear_color = {0, 0, 0, 1};
    debug_desc.usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc);

    rendering::TextureDeclHandle debug_decls[k_debug_target_count]{};
    for (uint32_t i = 0; i < eff_debug_count; ++i) {
        debug_decls[i] = fg.texture(eff_debug_targets[i].resource_name, debug_desc);
    }

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

    auto& ibl = ctx.world.ibl_resources();
    auto& ibl_pipes = ctx.world.ibl_pipelines();
    auto ibl_ready = ibl.is_ready();

    // IBL descriptor resources: when IBL is not ready the shader still samples
    // these views, so we point them at dedicated black fallbacks to guarantee
    // zero contribution (FallbackPool fills with white, which would blow out
    // the IBL term).
    auto fb_cube_decl = fg.find_texture("forward_ibl_fallback_cube");
    auto fb_2d_decl = fg.find_texture("forward_ibl_fallback_2d");
    INVARIANT(fb_cube_decl && fb_2d_decl);
    // Fallback WGPU views are read via direct-access on the compiled struct
    // because they're persistent textures uploaded on first compile (fallback
    // compiled pointers aren't available until FG compile() runs this frame).
    // Use external_view bindings with IBL's own views when ready, and leave
    // fallback paths to use managed texture bindings.
    auto ibl_prefiltered_view = ibl_ready ? ibl.prefiltered_env_view() : nullptr;
    auto ibl_env_cubemap_view = ibl_ready ? ibl.env_cubemap_view() : nullptr;
    auto ibl_irradiance_view = ibl_ready ? ibl.irradiance_view() : nullptr;
    auto ibl_brdf_lut_view = ibl_ready ? ibl_pipes.brdf_lut_view() : nullptr;

    // Descriptor 2: IBL
    auto ibl_sampler = fg.sampler(WGPUSamplerBindingType_Filtering, WGPUAddressMode_ClampToEdge,
                                  WGPUMipmapFilterMode_Linear);
    auto ibl_bld = descriptor(fg, ibl_desc_layout, "ibl_desc");
    if (ibl_ready) {
        ibl_bld.external_view(0, ibl_prefiltered_view)
            .external_view(1, ibl_irradiance_view)
            .external_view(2, ibl_brdf_lut_view);
    } else {
        ibl_bld.texture(0, fb_cube_decl).texture(1, fb_cube_decl).texture(2, fb_2d_decl);
    }
    auto desc2_decl = ibl_bld.sampler(3, ibl_sampler).build();

    // Contact shadow pass (after G-buffer, before forward lighting)
    auto* cs_pass = get_pass<rendering::ContactShadowPass>();
    PRECONDITION(cs_pass);
    auto cs_out = cs_pass->add_to_frame_graph(
        fg, ctx, {gbuf_out.depth, gbuf_out.normals, light_buf.handle(), light_buf.size()},
        fg.fallback_pool());

    // Descriptor 3: contact shadow (child-owned)
    PRECONDITION(cs_out.consumer_desc);

    // Skybox uniform buffer + descriptor
    rendering::BufferDesc skybox_buf_desc;
    skybox_buf_desc.size = sizeof(SkyboxUniforms);
    skybox_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto skybox_uniform_buf_decl = create_buffer(fg, skybox_buf_desc, "skybox_uniforms");

    auto skybox_bld = descriptor(fg, skybox_desc_layout, "skybox_desc")
                          .buffer(0, skybox_uniform_buf_decl, 0, sizeof(SkyboxUniforms));
    if (ibl_ready) {
        skybox_bld.external_view(1, ibl_env_cubemap_view);
    } else {
        skybox_bld.texture(1, fb_cube_decl);
    }
    auto skybox_desc_decl = skybox_bld.sampler(2, ibl_sampler).build();

    // Capture values for the execute lambda
    auto queue = ctx.queue;
    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    auto elapsed_time = ctx.time;
    auto camera_pos = ctx.camera_position;
    const auto& world = ctx.world;

    auto viewport_width = ctx.viewport_width;
    auto viewport_height = ctx.viewport_height;

    auto desc1_decl = shadow_out.consumer_desc;
    auto desc3_decl = cs_out.consumer_desc;

    auto pass_builder = fg.add_pass("forward").color(color_decl).read(shadow_out.shadow_array);
    if (cs_out.contact_shadow) {
        pass_builder.read(cs_out.contact_shadow);
    }
    for (uint32_t i = 0; i < eff_debug_count; ++i) {
        pass_builder.color(debug_decls[i]);
    }
    // Group 0 is dynamic (per-draw offsets); groups 1-3 are static (auto-set)
    pass_builder.descriptor(0, desc0_decl, rendering::dynamic_descriptor)
        .descriptor(1, desc1_decl)
        .descriptor(2, desc2_decl)
        .descriptor(3, desc3_decl);
    pass_builder.depth(depth_decl)
        .execute([=, &world](rendering::ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto objs = world.get_objects().span_raw();
            auto meshes_raw = world.get_meshes().span_raw();

            auto uniform_buf = exec.get(uniform_buf_decl).buffer;
            auto desc0 = exec.get(desc0_decl).bind_group;

            // Upload per-object uniforms
            {
                PTS_ZONE_NAMED("forward uniform upload");
                for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                    if (!objs[i].active) continue;
                    if (!objs[i].value.visible) continue;
                    const auto& obj = objs[i].value;
                    ForwardUniforms u{};
                    u.mvp = proj_mat * view_mat * obj.transform;
                    u.model = obj.transform;
                    u.camera_pos = camera_pos;
                    u.time = elapsed_time;
                    u.material_index = obj.material_index;
                    u.light_count = light_count;
                    u.viewport_size = {static_cast<float>(viewport_width),
                                       static_cast<float>(viewport_height)};
                    u.ibl_dome_modulation = ibl_ready ? dome_mod : glm::vec3{0.0f};
                    u.ibl_mip_count = rendering::k_prefilter_mip_count;
                    wgpuQueueWriteBuffer(queue, uniform_buf, i * k_uniform_align, &u, sizeof(u));
                }
            }

            // Upload uniforms for proxy light meshes
            {
                PTS_ZONE_NAMED("forward proxy light uniform upload");
                auto light_slots = world.get_lights().span_raw();
                uint32_t proxy_slot = object_count;
                for (uint32_t li = 0; li < static_cast<uint32_t>(light_slots.size()); ++li) {
                    if (!light_slots[li].active) continue;
                    if (light_slots[li].value.mesh_index == UINT32_MAX) continue;
                    if (!light_slots[li].value.visible) {
                        ++proxy_slot;
                        continue;
                    }
                    const auto& light = light_slots[li].value;
                    ForwardUniforms u{};
                    u.mvp = proj_mat * view_mat * light.transform;
                    u.model = light.transform;
                    u.camera_pos = camera_pos;
                    u.time = elapsed_time;
                    u.material_index = light.material_index;
                    u.light_count = light_count;
                    u.viewport_size = {static_cast<float>(viewport_width),
                                       static_cast<float>(viewport_height)};
                    u.ibl_dome_modulation = ibl_ready ? dome_mod : glm::vec3{0.0f};
                    u.ibl_mip_count = rendering::k_prefilter_mip_count;
                    wgpuQueueWriteBuffer(queue, uniform_buf, proxy_slot * k_uniform_align, &u,
                                         sizeof(u));
                    ++proxy_slot;
                }
            }

            // Upload skybox uniforms
            {
                auto skybox_buf = exec.get(skybox_uniform_buf_decl).buffer;
                SkyboxUniforms sky_u{};
                sky_u.inv_vp = glm::inverse(proj_mat * view_mat);
                sky_u.camera_pos = camera_pos;
                sky_u.dome_modulation = ibl_ready ? dome_mod : glm::vec3{0.0f};
                wgpuQueueWriteBuffer(queue, skybox_buf, 0, &sky_u, sizeof(sky_u));
            }

            wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);

            for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                if (!objs[i].active) continue;
                if (!objs[i].value.visible) continue;
                uint32_t dyn_offset = i * k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, desc0, 1, &dyn_offset);
                const auto& mesh = meshes_raw[objs[i].value.mesh_index].value;
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh.vertex_buffer.handle(), 0,
                                                     mesh.vertex_buffer.size());
                wgpuRenderPassEncoderSetIndexBuffer(pass, mesh.index_buffer.handle(),
                                                    WGPUIndexFormat_Uint32, 0,
                                                    mesh.index_buffer.size());
                wgpuRenderPassEncoderDrawIndexed(pass, mesh.index_count, 1, 0, 0, 0);
            }

            // Draw light proxy meshes
            {
                auto light_slots = world.get_lights().span_raw();
                uint32_t proxy_idx = object_count;
                for (uint32_t li = 0; li < static_cast<uint32_t>(light_slots.size()); ++li) {
                    if (!light_slots[li].active) continue;
                    if (light_slots[li].value.mesh_index == UINT32_MAX) continue;
                    if (!light_slots[li].value.visible) {
                        ++proxy_idx;
                        continue;
                    }
                    uint32_t dyn_offset = proxy_idx * k_uniform_align;
                    wgpuRenderPassEncoderSetBindGroup(pass, 0, desc0, 1, &dyn_offset);
                    const auto& mesh = meshes_raw[light_slots[li].value.mesh_index].value;
                    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh.vertex_buffer.handle(), 0,
                                                         mesh.vertex_buffer.size());
                    wgpuRenderPassEncoderSetIndexBuffer(pass, mesh.index_buffer.handle(),
                                                        WGPUIndexFormat_Uint32, 0,
                                                        mesh.index_buffer.size());
                    wgpuRenderPassEncoderDrawIndexed(pass, mesh.index_count, 1, 0, 0, 0);
                    ++proxy_idx;
                }
            }

            // Skybox: draw fullscreen triangle after all geometry
            if (ibl_ready) {
                auto skybox_desc = exec.get(skybox_desc_decl).bind_group;
                wgpuRenderPassEncoderSetPipeline(pass, skybox_pipeline_handle);
                wgpuRenderPassEncoderSetBindGroup(pass, 0, skybox_desc, 0, nullptr);
                wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            }
        });

    // Post-pass: SSAO
    rendering::TextureDeclHandle ssao_decl;
    if (auto* ssao = get_pass<rendering::SSAOPass>()) {
        auto ssao_out = ssao->add_to_frame_graph(fg, ctx, {gbuf_out.depth, gbuf_out.normals},
                                                 fg.fallback_pool());
        if (ssao_out.ssao) ssao_decl = ssao_out.ssao;
    }

    return {color_decl, depth_decl, ssao_decl};
}
