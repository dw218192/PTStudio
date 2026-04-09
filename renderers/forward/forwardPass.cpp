#include "forwardPass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/contactShadowPass.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/gbufferPass.h>
#include <core/rendering/iblResources.h>
#include <core/rendering/outputLayout.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/rendererRegistry.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/ssaoPass.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <renderers/forward/generated/shader_metadata.h>

#include <glm/glm.hpp>

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

ForwardPass::~ForwardPass() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->descriptor_layout) wgpuBindGroupLayoutRelease(ready->descriptor_layout);
        if (ready->ibl_desc_layout) wgpuBindGroupLayoutRelease(ready->ibl_desc_layout);
        if (ready->ibl_sampler) wgpuSamplerRelease(ready->ibl_sampler);
        if (ready->fallback_cube_view) wgpuTextureViewRelease(ready->fallback_cube_view);
        if (ready->fallback_cube_tex) wgpuTextureRelease(ready->fallback_cube_tex);
        if (ready->fallback_2d_view) wgpuTextureViewRelease(ready->fallback_2d_view);
        if (ready->fallback_2d_tex) wgpuTextureRelease(ready->fallback_2d_tex);
        if (ready->skybox_desc_layout) wgpuBindGroupLayoutRelease(ready->skybox_desc_layout);
    }
}

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

auto ForwardPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

auto ForwardPass::renderer_debug_targets() const noexcept
    -> std::pair<const DebugTarget*, uint32_t> {
    return {k_debug_targets, k_debug_target_count};
}

void ForwardPass::do_renderer_setup(const webgpu::Device& device) {
    // Release existing state for re-entry (hot-reload)
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->descriptor_layout) wgpuBindGroupLayoutRelease(ready->descriptor_layout);
        if (ready->ibl_desc_layout) wgpuBindGroupLayoutRelease(ready->ibl_desc_layout);
        if (ready->ibl_sampler) wgpuSamplerRelease(ready->ibl_sampler);
        if (ready->fallback_cube_view) wgpuTextureViewRelease(ready->fallback_cube_view);
        if (ready->fallback_cube_tex) wgpuTextureRelease(ready->fallback_cube_tex);
        if (ready->fallback_2d_view) wgpuTextureViewRelease(ready->fallback_2d_view);
        if (ready->fallback_2d_tex) wgpuTextureRelease(ready->fallback_2d_tex);
        if (ready->skybox_desc_layout) wgpuBindGroupLayoutRelease(ready->skybox_desc_layout);
    }

    auto* shadow = get_pass<rendering::ShadowMapPass>();
    auto* cs = get_pass<rendering::ContactShadowPass>();
    PRECONDITION_MSG(shadow && shadow->is_ready(),
                     "ShadowMapPass must be ready before ForwardPass");
    PRECONDITION_MSG(cs && cs->is_ready(), "ContactShadowPass must be ready before ForwardPass");

    auto [dbg_targets_setup, dbg_count_setup] = effective_debug_targets();
    auto shader_src = load_pass_shader("renderers/forward/generated/shaders/forward.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    // Create descriptor 0 layout via OutputSlot API
    auto bg0_internal = create_output_layout(
        device,
        {OutputSlot::uniform(sizeof(ForwardUniforms))
             .dynamic()
             .visibility(
                 static_cast<WGPUShaderStage>(WGPUShaderStage_Vertex | WGPUShaderStage_Fragment)),
         OutputSlot::storage(), OutputSlot::storage(),
         OutputSlot::texture(WGPUTextureFormat_RGBA32Float),
         OutputSlot::texture(WGPUTextureFormat_RG32Float),
         OutputSlot::sampler(WGPUSamplerBindingType_Filtering),
         OutputSlot::texture(WGPUTextureFormat_RGBA8Unorm, WGPUTextureViewDimension_2DArray),
         OutputSlot::sampler(WGPUSamplerBindingType_Filtering)});
    auto descriptor_layout = bg0_internal.layout;
    bg0_internal.layout = nullptr;
    bg0_internal.release();

    // --- IBL descriptor layout (group 2) via OutputSlot API ---
    auto ibl_internal = create_output_layout(
        device, {OutputSlot::texture(WGPUTextureFormat_RGBA8Unorm, WGPUTextureViewDimension_Cube),
                 OutputSlot::texture(WGPUTextureFormat_RGBA8Unorm, WGPUTextureViewDimension_Cube),
                 OutputSlot::texture(WGPUTextureFormat_RGBA8Unorm),
                 OutputSlot::sampler(WGPUSamplerBindingType_Filtering)});
    auto ibl_desc_layout = ibl_internal.layout;
    ibl_internal.layout = nullptr;
    ibl_internal.release();

    // --- IBL sampler ---
    WGPUSamplerDescriptor ibl_samp_desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    ibl_samp_desc.magFilter = WGPUFilterMode_Linear;
    ibl_samp_desc.minFilter = WGPUFilterMode_Linear;
    ibl_samp_desc.mipmapFilter = WGPUMipmapFilterMode_Linear;
    ibl_samp_desc.addressModeU = WGPUAddressMode_ClampToEdge;
    ibl_samp_desc.addressModeV = WGPUAddressMode_ClampToEdge;
    ibl_samp_desc.addressModeW = WGPUAddressMode_ClampToEdge;
    auto ibl_sampler = wgpuDeviceCreateSampler(device.handle(), &ibl_samp_desc);

    // --- 1x1 black fallback textures for IBL when not yet ready ---
    WGPUTextureDescriptor fb_cube_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    fb_cube_desc.size = {1, 1, 6};
    fb_cube_desc.format = WGPUTextureFormat_RGBA8Unorm;
    fb_cube_desc.usage = WGPUTextureUsage_TextureBinding;
    fb_cube_desc.dimension = WGPUTextureDimension_2D;
    fb_cube_desc.mipLevelCount = 1;
    auto fallback_cube_tex = wgpuDeviceCreateTexture(device.handle(), &fb_cube_desc);

    WGPUTextureViewDescriptor fb_cube_view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    fb_cube_view_desc.dimension = WGPUTextureViewDimension_Cube;
    fb_cube_view_desc.format = WGPUTextureFormat_RGBA8Unorm;
    fb_cube_view_desc.arrayLayerCount = 6;
    fb_cube_view_desc.mipLevelCount = 1;
    auto fallback_cube_view = wgpuTextureCreateView(fallback_cube_tex, &fb_cube_view_desc);

    WGPUTextureDescriptor fb_2d_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    fb_2d_desc.size = {1, 1, 1};
    fb_2d_desc.format = WGPUTextureFormat_RGBA8Unorm;
    fb_2d_desc.usage = WGPUTextureUsage_TextureBinding;
    fb_2d_desc.dimension = WGPUTextureDimension_2D;
    fb_2d_desc.mipLevelCount = 1;
    auto fallback_2d_tex = wgpuDeviceCreateTexture(device.handle(), &fb_2d_desc);

    WGPUTextureViewDescriptor fb_2d_view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    fb_2d_view_desc.dimension = WGPUTextureViewDimension_2D;
    fb_2d_view_desc.format = WGPUTextureFormat_RGBA8Unorm;
    fb_2d_view_desc.arrayLayerCount = 1;
    fb_2d_view_desc.mipLevelCount = 1;
    auto fallback_2d_view = wgpuTextureCreateView(fallback_2d_tex, &fb_2d_view_desc);

    // --- Pipeline layout with 4 descriptors (child passes own groups 1 and 3) ---
    WGPUBindGroupLayout bgls[4] = {descriptor_layout, shadow->consumer_layout(), ibl_desc_layout,
                                   cs->consumer_layout()};
    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 4;
    pl_desc.bindGroupLayouts = bgls;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

    auto builder = webgpu::RenderPipelineBuilder(device)
                       .shader(shader)
                       .color_format(WGPUTextureFormat_RGBA16Float, 0)
                       .depth_format(WGPUTextureFormat_Depth32Float)
                       .depth_write(true)
                       .depth_compare(WGPUCompareFunction_LessEqual)
                       .cull_mode(WGPUCullMode_Back)
                       .pipeline_layout(pipeline_layout)
                       .vertex_layout<forward_shader::VertexLayout>();
    for (uint32_t i = 0; i < dbg_count_setup; ++i) {
        builder.color_format(WGPUTextureFormat_RGBA8Unorm, i + 1);
    }
    auto pipeline = builder.build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    // --- Skybox pipeline ---
    auto skybox_shader_src =
        get_shader_loader().load("renderers/forward/generated/shaders/skybox.wgsl");
    auto skybox_shader = device.create_shader_module_from_source(skybox_shader_src);

    // Skybox BGL via OutputSlot API: uniform buffer (Vert|Frag), cube texture (Frag), sampler
    // (Frag)
    auto skybox_internal = create_output_layout(
        device, {OutputSlot::uniform(sizeof(SkyboxUniforms))
                     .visibility(static_cast<WGPUShaderStage>(WGPUShaderStage_Vertex |
                                                              WGPUShaderStage_Fragment)),
                 OutputSlot::texture(WGPUTextureFormat_RGBA8Unorm, WGPUTextureViewDimension_Cube),
                 OutputSlot::sampler(WGPUSamplerBindingType_Filtering)});
    auto skybox_desc_layout = skybox_internal.layout;
    skybox_internal.layout = nullptr;
    skybox_internal.release();

    WGPUPipelineLayoutDescriptor skybox_pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    skybox_pl_desc.bindGroupLayoutCount = 1;
    skybox_pl_desc.bindGroupLayouts = &skybox_desc_layout;
    auto skybox_pl = wgpuDeviceCreatePipelineLayout(device.handle(), &skybox_pl_desc);

    auto skybox_builder = webgpu::RenderPipelineBuilder(device)
                              .shader(skybox_shader)
                              .color_format(WGPUTextureFormat_RGBA16Float, 0)
                              .depth_format(WGPUTextureFormat_Depth32Float)
                              .depth_write(false)
                              .depth_compare(WGPUCompareFunction_LessEqual)
                              .cull_mode(WGPUCullMode_None)
                              .pipeline_layout(skybox_pl);
    for (uint32_t i = 0; i < dbg_count_setup; ++i) {
        skybox_builder.color_format(WGPUTextureFormat_RGBA8Unorm, i + 1)
            .write_mask(WGPUColorWriteMask_None, i + 1);
    }
    auto skybox_pipeline = skybox_builder.build();

    wgpuPipelineLayoutRelease(skybox_pl);

    rendering::LtcTextures ltc;
    ltc.init(device);

    m_state = Ready{
        std::move(shader),  std::move(pipeline),      descriptor_layout,
        std::move(ltc),     ibl_desc_layout,          ibl_sampler,
        fallback_cube_tex,  fallback_cube_view,       fallback_2d_tex,
        fallback_2d_view,   std::move(skybox_shader), std::move(skybox_pipeline),
        skybox_desc_layout,
    };
}

ForwardPass::HdrOutputs ForwardPass::do_add_to_frame_graph(rendering::FrameGraph& fg,
                                                           const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());

    // Pre-passes: G-buffer (depth + normals) and shadow maps
    rendering::GBufferPass::Outputs gbuf_out;
    if (auto* gbuf = get_pass<rendering::GBufferPass>(); gbuf && gbuf->is_ready())
        gbuf_out = gbuf->add_to_frame_graph(fg, ctx, {});

    rendering::ShadowMapPass::Outputs shadow_out{};
    if (auto* shadow = get_pass<rendering::ShadowMapPass>(); shadow && shadow->is_ready())
        shadow_out = shadow->add_to_frame_graph(fg, ctx, {});

    auto& ready = std::get<Ready>(m_state);

    auto objects = ctx.world.get_objects();
    auto object_count = static_cast<uint32_t>(objects.size());

    // Count proxy lights (lights with active mesh proxies) for uniform buffer sizing
    auto all_lights = ctx.world.get_lights();
    uint32_t proxy_light_count = 0;
    for (uint32_t li = 0; li < static_cast<uint32_t>(all_lights.size()); ++li) {
        if (!all_lights[li].active()) continue;
        if (all_lights[li]->mesh_index == UINT32_MAX) continue;
        ++proxy_light_count;
    }

    uint32_t total_slots = object_count + proxy_light_count;

    // Import external buffers from RenderWorld
    auto& light_buf = ctx.world.light_buffer();
    auto& mat_buf = ctx.world.material_buffer();
    auto light_count = ctx.world.gpu_light_count();
    auto light_buf_handle = import_buffer(fg, light_buf.handle(), light_buf.size(), "world_lights");
    auto mat_buf_handle = import_buffer(fg, mat_buf.handle(), mat_buf.size(), "world_materials");

    auto scene_tex_view = ctx.world.texture_array_view();
    auto scene_tex_sampler = ctx.world.texture_sampler();

    // Managed uniform buffer
    uint64_t uniform_needed =
        std::max(uint64_t(1), static_cast<uint64_t>(total_slots)) * k_uniform_align;
    rendering::BufferDesc uniform_buf_desc;
    uniform_buf_desc.size = uniform_needed;
    uniform_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_handle = create_buffer(fg, uniform_buf_desc, "uniforms");

    // Descriptor 0: materials, lights, uniforms, LTC, scene textures
    auto bg0_handle = descriptor(fg, ready.descriptor_layout, "bg0")
                          .buffer(0, uniform_buf_handle, 0, sizeof(ForwardUniforms))
                          .buffer(1, mat_buf_handle)
                          .buffer(2, light_buf_handle)
                          .external_view(3, ready.ltc_textures.mat_view())
                          .external_view(4, ready.ltc_textures.amp_view())
                          .sampler(5, ready.ltc_textures.sampler())
                          .external_view(6, scene_tex_view)
                          .sampler(7, scene_tex_sampler)
                          .build();

    // Descriptor 1: shadow (child-owned)
    PRECONDITION(shadow_out.consumer_desc.is_valid());

    rendering::TextureDesc color_desc;
    color_desc.width = ctx.viewport_width;
    color_desc.height = ctx.viewport_height;
    color_desc.format = WGPUTextureFormat_RGBA16Float;
    color_desc.clear_color = {0.15, 0.15, 0.18, 1.0};

    rendering::TextureDesc depth_desc;
    depth_desc.width = ctx.viewport_width;
    depth_desc.height = ctx.viewport_height;
    depth_desc.format = WGPUTextureFormat_Depth32Float;

    auto color = create_texture(fg, color_desc, "color");
    auto depth =
        gbuf_out.depth.is_valid() ? gbuf_out.depth : create_texture(fg, depth_desc, "depth");

    auto [eff_debug_targets, eff_debug_count] = effective_debug_targets();

    rendering::TextureDesc debug_desc;
    debug_desc.width = ctx.viewport_width;
    debug_desc.height = ctx.viewport_height;
    debug_desc.format = WGPUTextureFormat_RGBA8Unorm;
    debug_desc.clear_color = {0, 0, 0, 1};
    debug_desc.usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc);

    rendering::ResourceHandle debug_handles[k_debug_target_count];
    for (uint32_t i = 0; i < eff_debug_count; ++i) {
        debug_handles[i] = fg.find_or_create(eff_debug_targets[i].resource_name, debug_desc);
    }

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

    auto& ibl = ctx.world.ibl_resources();
    auto& ibl_pipes = ctx.world.ibl_pipelines();
    auto ibl_ready = ibl.is_ready();

    // IBL descriptor resources (use fallback textures when IBL not ready)
    auto ibl_prefiltered_view = ibl_ready ? ibl.prefiltered_env_view() : ready.fallback_cube_view;
    auto ibl_env_cubemap_view = ibl_ready ? ibl.env_cubemap_view() : ready.fallback_cube_view;
    auto ibl_irradiance_view = ibl_ready ? ibl.irradiance_view() : ready.fallback_cube_view;
    auto ibl_brdf_lut_view = ibl_ready ? ibl_pipes.brdf_lut_view() : ready.fallback_2d_view;

    // Descriptor 2: IBL
    auto bg2_handle = descriptor(fg, ready.ibl_desc_layout, "ibl_bg")
                          .external_view(0, ibl_prefiltered_view)
                          .external_view(1, ibl_irradiance_view)
                          .external_view(2, ibl_brdf_lut_view)
                          .sampler(3, ready.ibl_sampler)
                          .build();

    // Contact shadow pass (after G-buffer, before forward lighting)
    auto* cs_pass = get_pass<rendering::ContactShadowPass>();
    PRECONDITION(cs_pass && cs_pass->is_ready());
    auto cs_out = cs_pass->add_to_frame_graph(
        fg, ctx, {gbuf_out.depth, gbuf_out.normals, light_buf.handle(), light_buf.size()},
        fg.fallback_pool());

    // Bind group 3: contact shadow (child-owned)
    PRECONDITION(cs_out.consumer_desc.is_valid());

    // Skybox uniform buffer + descriptor
    rendering::BufferDesc skybox_buf_desc;
    skybox_buf_desc.size = sizeof(SkyboxUniforms);
    skybox_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto skybox_uniform_buf_handle = create_buffer(fg, skybox_buf_desc, "skybox_uniforms");

    auto skybox_bg_handle = descriptor(fg, ready.skybox_desc_layout, "skybox_bg")
                                .buffer(0, skybox_uniform_buf_handle, 0, sizeof(SkyboxUniforms))
                                .external_view(1, ibl_env_cubemap_view)
                                .sampler(2, ready.ibl_sampler)
                                .build();

    // Capture values for the execute lambda
    auto queue = ctx.queue;
    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    auto elapsed_time = ctx.time;
    auto camera_pos = ctx.camera_position;
    auto* pipeline_handle = ready.pipeline.handle();
    auto skybox_pipeline_handle = ready.skybox_pipeline.handle();
    const auto& world = ctx.world;

    auto viewport_width = ctx.viewport_width;
    auto viewport_height = ctx.viewport_height;

    auto bg1_handle = shadow_out.consumer_desc;
    auto bg3_handle = cs_out.consumer_desc;

    auto pass_builder = fg.add_pass("forward").color(color).read(shadow_out.shadow_array);
    if (cs_out.contact_shadow.is_valid()) {
        pass_builder.read(cs_out.contact_shadow);
    }
    for (uint32_t i = 0; i < eff_debug_count; ++i) {
        pass_builder.color(debug_handles[i]);
    }
    // Group 0 is dynamic (per-draw offsets); groups 1-3 are static (auto-set)
    pass_builder.descriptor(0, bg0_handle, rendering::dynamic_descriptor)
        .descriptor(1, bg1_handle)
        .descriptor(2, bg2_handle)
        .descriptor(3, bg3_handle);
    pass_builder.depth(depth).execute([=, &fg, &world](WGPURenderPassEncoder pass) {
        auto objs = world.get_objects();
        auto meshes = world.get_meshes();

        auto uniform_buf = fg.get_buffer_ref(uniform_buf_handle).handle();
        auto bg0 = fg.get_descriptor_ref(bg0_handle).handle();

        // Upload per-object uniforms
        {
            PTS_ZONE_NAMED("forward uniform upload");
            for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
                if (!objs[i].active()) continue;
                if (!objs[i]->visible) continue;
                const auto& obj = objs[i];
                ForwardUniforms u{};
                u.mvp = proj_mat * view_mat * obj->transform;
                u.model = obj->transform;
                u.camera_pos = camera_pos;
                u.time = elapsed_time;
                u.material_index = obj->material_index;
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
            auto light_slots = world.get_lights();
            uint32_t proxy_slot = object_count;
            for (uint32_t li = 0; li < static_cast<uint32_t>(light_slots.size()); ++li) {
                if (!light_slots[li].active()) continue;
                if (light_slots[li]->mesh_index == UINT32_MAX) continue;
                if (!light_slots[li]->visible) {
                    ++proxy_slot;
                    continue;
                }
                ForwardUniforms u{};
                u.mvp = proj_mat * view_mat * light_slots[li]->transform;
                u.model = light_slots[li]->transform;
                u.camera_pos = camera_pos;
                u.time = elapsed_time;
                u.material_index = light_slots[li]->material_index;
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
            auto skybox_buf = fg.get_buffer_ref(skybox_uniform_buf_handle).handle();
            SkyboxUniforms sky_u{};
            sky_u.inv_vp = glm::inverse(proj_mat * view_mat);
            sky_u.camera_pos = camera_pos;
            sky_u.dome_modulation = ibl_ready ? dome_mod : glm::vec3{0.0f};
            wgpuQueueWriteBuffer(queue, skybox_buf, 0, &sky_u, sizeof(sky_u));
        }

        wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);

        for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
            if (!objs[i].active()) continue;
            if (!objs[i]->visible) continue;
            uint32_t dyn_offset = i * k_uniform_align;
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bg0, 1, &dyn_offset);
            const auto& mesh = meshes[objs[i]->mesh_index];
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh->vertex_buffer.handle(), 0,
                                                 mesh->vertex_buffer.size());
            wgpuRenderPassEncoderSetIndexBuffer(pass, mesh->index_buffer.handle(),
                                                WGPUIndexFormat_Uint32, 0,
                                                mesh->index_buffer.size());
            wgpuRenderPassEncoderDrawIndexed(pass, mesh->index_count, 1, 0, 0, 0);
        }

        // Draw light proxy meshes
        {
            auto light_slots = world.get_lights();
            uint32_t proxy_idx = object_count;
            for (uint32_t li = 0; li < static_cast<uint32_t>(light_slots.size()); ++li) {
                if (!light_slots[li].active()) continue;
                if (light_slots[li]->mesh_index == UINT32_MAX) continue;
                if (!light_slots[li]->visible) {
                    ++proxy_idx;
                    continue;
                }
                uint32_t dyn_offset = proxy_idx * k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, bg0, 1, &dyn_offset);
                const auto& mesh = meshes[light_slots[li]->mesh_index];
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh->vertex_buffer.handle(), 0,
                                                     mesh->vertex_buffer.size());
                wgpuRenderPassEncoderSetIndexBuffer(pass, mesh->index_buffer.handle(),
                                                    WGPUIndexFormat_Uint32, 0,
                                                    mesh->index_buffer.size());
                wgpuRenderPassEncoderDrawIndexed(pass, mesh->index_count, 1, 0, 0, 0);
                ++proxy_idx;
            }
        }

        // Skybox: draw fullscreen triangle after all geometry
        if (ibl_ready) {
            auto skybox_bg = fg.get_descriptor_ref(skybox_bg_handle).handle();
            wgpuRenderPassEncoderSetPipeline(pass, skybox_pipeline_handle);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, skybox_bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        }
    });

    // Post-pass: SSAO
    std::optional<rendering::TextureHandle> ssao_handle;
    if (auto* ssao = get_pass<rendering::SSAOPass>(); ssao && ssao->is_ready()) {
        auto ssao_out = ssao->add_to_frame_graph(fg, ctx, {gbuf_out.depth, gbuf_out.normals});
        if (ssao_out.ssao.is_valid()) ssao_handle = rendering::TextureHandle{ssao_out.ssao.index};
    }

    return {color, rendering::TextureHandle{depth}, ssao_handle};
}
