#include "forwardPass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/gbufferPass.h>
#include <core/rendering/iblResources.h>
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

REGISTER_RENDERER("Forward", ForwardPass);

ForwardPass::ForwardPass(const rendering::ShaderLoader& sl) : IRenderer(sl) {
    add_pass<rendering::GBufferPass>(sl);
    add_pass<rendering::ShadowMapPass>(sl);
    add_pass<rendering::SSAOPass>(sl);
}

struct ForwardUniforms {
    glm::mat4 mvp;
    glm::mat4 model;
    glm::vec3 camera_pos;
    float time;
    uint32_t material_index;
    uint32_t light_count;
    uint32_t _pad[2];
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
        if (ready->bind_group_layout) wgpuBindGroupLayoutRelease(ready->bind_group_layout);
        if (ready->shadow_recv_bgl) wgpuBindGroupLayoutRelease(ready->shadow_recv_bgl);
        if (ready->shadow_sampler) wgpuSamplerRelease(ready->shadow_sampler);
        if (ready->ibl_bgl) wgpuBindGroupLayoutRelease(ready->ibl_bgl);
        if (ready->ibl_sampler) wgpuSamplerRelease(ready->ibl_sampler);
        if (ready->fallback_cube_view) wgpuTextureViewRelease(ready->fallback_cube_view);
        if (ready->fallback_cube_tex) wgpuTextureRelease(ready->fallback_cube_tex);
        if (ready->fallback_2d_view) wgpuTextureViewRelease(ready->fallback_2d_view);
        if (ready->fallback_2d_tex) wgpuTextureRelease(ready->fallback_2d_tex);
        if (ready->skybox_bgl) wgpuBindGroupLayoutRelease(ready->skybox_bgl);
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

auto ForwardPass::debug_targets() const noexcept -> std::pair<const DebugTarget*, uint32_t> {
    return {k_debug_targets, k_debug_target_count};
}

void ForwardPass::do_renderer_setup(const webgpu::Device& device) {
    // Release existing state for re-entry (hot-reload)
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group_layout) wgpuBindGroupLayoutRelease(ready->bind_group_layout);
        if (ready->shadow_recv_bgl) wgpuBindGroupLayoutRelease(ready->shadow_recv_bgl);
        if (ready->shadow_sampler) wgpuSamplerRelease(ready->shadow_sampler);
        if (ready->ibl_bgl) wgpuBindGroupLayoutRelease(ready->ibl_bgl);
        if (ready->ibl_sampler) wgpuSamplerRelease(ready->ibl_sampler);
        if (ready->fallback_cube_view) wgpuTextureViewRelease(ready->fallback_cube_view);
        if (ready->fallback_cube_tex) wgpuTextureRelease(ready->fallback_cube_tex);
        if (ready->fallback_2d_view) wgpuTextureViewRelease(ready->fallback_2d_view);
        if (ready->fallback_2d_tex) wgpuTextureRelease(ready->fallback_2d_tex);
        if (ready->skybox_bgl) wgpuBindGroupLayoutRelease(ready->skybox_bgl);
    }

    auto [dbg_targets_setup, dbg_count_setup] = effective_debug_targets();
    auto shader_src = load_pass_shader("renderers/forward/generated/shaders/forward.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    // Create bind group 0 layout: binding 0 = uniform (dynamic), 1 = storage (materials),
    // 2 = storage (lights), 3 = texture (LTC mat), 4 = texture (LTC amp), 5 = sampler (LTC),
    // 6 = texture array (scene textures), 7 = sampler (scene textures)
    WGPUBindGroupLayoutEntry entries[8] = {};

    entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[0].binding = 0;
    entries[0].visibility =
        static_cast<WGPUShaderStage>(WGPUShaderStage_Vertex | WGPUShaderStage_Fragment);
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    entries[0].buffer.hasDynamicOffset = true;
    entries[0].buffer.minBindingSize = sizeof(ForwardUniforms);

    entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[1].buffer.minBindingSize = 0;

    entries[2] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Fragment;
    entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[2].buffer.minBindingSize = 0;

    entries[3] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[3].binding = 3;
    entries[3].visibility = WGPUShaderStage_Fragment;
    entries[3].texture.sampleType = WGPUTextureSampleType_Float;
    entries[3].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[3].texture.multisampled = false;

    entries[4] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[4].binding = 4;
    entries[4].visibility = WGPUShaderStage_Fragment;
    entries[4].texture.sampleType = WGPUTextureSampleType_Float;
    entries[4].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[4].texture.multisampled = false;

    entries[5] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[5].binding = 5;
    entries[5].visibility = WGPUShaderStage_Fragment;
    entries[5].sampler.type = WGPUSamplerBindingType_Filtering;

    entries[6] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[6].binding = 6;
    entries[6].visibility = WGPUShaderStage_Fragment;
    entries[6].texture.sampleType = WGPUTextureSampleType_Float;
    entries[6].texture.viewDimension = WGPUTextureViewDimension_2DArray;
    entries[6].texture.multisampled = false;

    entries[7] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[7].binding = 7;
    entries[7].visibility = WGPUShaderStage_Fragment;
    entries[7].sampler.type = WGPUSamplerBindingType_Filtering;

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = 8;
    bgl_desc.entries = entries;
    auto bind_group_layout = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);

    // --- Shadow receiver bind group layout (group 1) ---
    WGPUBindGroupLayoutEntry shadow_entries[3] = {};

    // binding 0: ShadowInfo storage buffer (read-only, one per light)
    shadow_entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    shadow_entries[0].binding = 0;
    shadow_entries[0].visibility = WGPUShaderStage_Fragment;
    shadow_entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    shadow_entries[0].buffer.minBindingSize = 80;  // sizeof(ShadowInfo)

    // binding 1: shadow depth texture array
    shadow_entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    shadow_entries[1].binding = 1;
    shadow_entries[1].visibility = WGPUShaderStage_Fragment;
    shadow_entries[1].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
    shadow_entries[1].texture.viewDimension = WGPUTextureViewDimension_2DArray;

    // binding 2: sampler
    shadow_entries[2] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    shadow_entries[2].binding = 2;
    shadow_entries[2].visibility = WGPUShaderStage_Fragment;
    shadow_entries[2].sampler.type = WGPUSamplerBindingType_NonFiltering;

    WGPUBindGroupLayoutDescriptor shadow_bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    shadow_bgl_desc.entryCount = 3;
    shadow_bgl_desc.entries = shadow_entries;
    auto shadow_recv_bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &shadow_bgl_desc);

    // --- Comparison sampler ---
    WGPUSamplerDescriptor sampler_desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    sampler_desc.magFilter = WGPUFilterMode_Nearest;
    sampler_desc.minFilter = WGPUFilterMode_Nearest;
    sampler_desc.addressModeU = WGPUAddressMode_ClampToEdge;
    sampler_desc.addressModeV = WGPUAddressMode_ClampToEdge;
    sampler_desc.addressModeW = WGPUAddressMode_ClampToEdge;
    auto shadow_sampler = wgpuDeviceCreateSampler(device.handle(), &sampler_desc);

    // --- IBL bind group layout (group 2) ---
    WGPUBindGroupLayoutEntry ibl_entries[4] = {};

    // binding 0: prefiltered env cubemap
    ibl_entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    ibl_entries[0].binding = 0;
    ibl_entries[0].visibility = WGPUShaderStage_Fragment;
    ibl_entries[0].texture.sampleType = WGPUTextureSampleType_Float;
    ibl_entries[0].texture.viewDimension = WGPUTextureViewDimension_Cube;

    // binding 1: irradiance cubemap
    ibl_entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    ibl_entries[1].binding = 1;
    ibl_entries[1].visibility = WGPUShaderStage_Fragment;
    ibl_entries[1].texture.sampleType = WGPUTextureSampleType_Float;
    ibl_entries[1].texture.viewDimension = WGPUTextureViewDimension_Cube;

    // binding 2: BRDF LUT
    ibl_entries[2] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    ibl_entries[2].binding = 2;
    ibl_entries[2].visibility = WGPUShaderStage_Fragment;
    ibl_entries[2].texture.sampleType = WGPUTextureSampleType_Float;
    ibl_entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

    // binding 3: sampler
    ibl_entries[3] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    ibl_entries[3].binding = 3;
    ibl_entries[3].visibility = WGPUShaderStage_Fragment;
    ibl_entries[3].sampler.type = WGPUSamplerBindingType_Filtering;

    WGPUBindGroupLayoutDescriptor ibl_bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    ibl_bgl_desc.entryCount = 4;
    ibl_bgl_desc.entries = ibl_entries;
    auto ibl_bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &ibl_bgl_desc);

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

    // --- Pipeline layout with 3 bind groups ---
    WGPUBindGroupLayout bgls[3] = {bind_group_layout, shadow_recv_bgl, ibl_bgl};
    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 3;
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

    // Skybox BGL: uniform buffer (Vert|Frag), cube texture (Frag), sampler (Frag)
    WGPUBindGroupLayoutEntry skybox_bgl_entries[3] = {};

    skybox_bgl_entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    skybox_bgl_entries[0].binding = 0;
    skybox_bgl_entries[0].visibility =
        static_cast<WGPUShaderStage>(WGPUShaderStage_Vertex | WGPUShaderStage_Fragment);
    skybox_bgl_entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    skybox_bgl_entries[0].buffer.minBindingSize = sizeof(SkyboxUniforms);

    skybox_bgl_entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    skybox_bgl_entries[1].binding = 1;
    skybox_bgl_entries[1].visibility = WGPUShaderStage_Fragment;
    skybox_bgl_entries[1].texture.sampleType = WGPUTextureSampleType_Float;
    skybox_bgl_entries[1].texture.viewDimension = WGPUTextureViewDimension_Cube;

    skybox_bgl_entries[2] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    skybox_bgl_entries[2].binding = 2;
    skybox_bgl_entries[2].visibility = WGPUShaderStage_Fragment;
    skybox_bgl_entries[2].sampler.type = WGPUSamplerBindingType_Filtering;

    WGPUBindGroupLayoutDescriptor skybox_bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    skybox_bgl_desc.entryCount = 3;
    skybox_bgl_desc.entries = skybox_bgl_entries;
    auto skybox_bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &skybox_bgl_desc);

    WGPUPipelineLayoutDescriptor skybox_pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    skybox_pl_desc.bindGroupLayoutCount = 1;
    skybox_pl_desc.bindGroupLayouts = &skybox_bgl;
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
        std::move(shader),
        std::move(pipeline),
        bind_group_layout,
        std::move(ltc),
        shadow_recv_bgl,
        shadow_sampler,
        ibl_bgl,
        ibl_sampler,
        fallback_cube_tex,
        fallback_cube_view,
        fallback_2d_tex,
        fallback_2d_view,
        std::move(skybox_shader),
        std::move(skybox_pipeline),
        skybox_bgl,
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

    // Bind group 0: materials, lights, uniforms, LTC, scene textures
    rendering::BindGroupEntry bg0_entries[8] = {};
    bg0_entries[0].binding = 0;
    bg0_entries[0].buffer = uniform_buf_handle;
    bg0_entries[0].buffer_size = sizeof(ForwardUniforms);

    bg0_entries[1].binding = 1;
    bg0_entries[1].buffer = mat_buf_handle;

    bg0_entries[2].binding = 2;
    bg0_entries[2].buffer = light_buf_handle;

    bg0_entries[3].binding = 3;
    bg0_entries[3].external_view = ready.ltc_textures.mat_view();

    bg0_entries[4].binding = 4;
    bg0_entries[4].external_view = ready.ltc_textures.amp_view();

    bg0_entries[5].binding = 5;
    bg0_entries[5].sampler = ready.ltc_textures.sampler();

    bg0_entries[6].binding = 6;
    bg0_entries[6].external_view = scene_tex_view;

    bg0_entries[7].binding = 7;
    bg0_entries[7].sampler = scene_tex_sampler;

    rendering::BindGroupDesc bg0_desc;
    bg0_desc.layout = ready.bind_group_layout;
    bg0_desc.entries.assign(std::begin(bg0_entries), std::end(bg0_entries));
    auto bg0_handle = create_bind_group(fg, std::move(bg0_desc), "bg0");

    // Bind group 1: shadow
    PRECONDITION(shadow_out.shadow_array.is_valid());
    PRECONDITION(shadow_out.shadow_info.is_valid());

    rendering::BindGroupEntry bg1_entries[3] = {};
    bg1_entries[0].binding = 0;
    bg1_entries[0].buffer = shadow_out.shadow_info;

    bg1_entries[1].binding = 1;
    bg1_entries[1].texture = shadow_out.shadow_array;

    bg1_entries[2].binding = 2;
    bg1_entries[2].sampler = ready.shadow_sampler;

    rendering::BindGroupDesc bg1_desc;
    bg1_desc.layout = ready.shadow_recv_bgl;
    bg1_desc.entries.assign(std::begin(bg1_entries), std::end(bg1_entries));
    auto bg1_handle = create_bind_group(fg, std::move(bg1_desc), "shadow_bg");

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

    // IBL bind group resources (use fallback textures when IBL not ready)
    auto ibl_prefiltered_view = ibl_ready ? ibl.prefiltered_env_view() : ready.fallback_cube_view;
    auto ibl_env_cubemap_view = ibl_ready ? ibl.env_cubemap_view() : ready.fallback_cube_view;
    auto ibl_irradiance_view = ibl_ready ? ibl.irradiance_view() : ready.fallback_cube_view;
    auto ibl_brdf_lut_view = ibl_ready ? ibl_pipes.brdf_lut_view() : ready.fallback_2d_view;

    // Bind group 2: IBL
    rendering::BindGroupEntry bg2_entries[4] = {};
    bg2_entries[0].binding = 0;
    bg2_entries[0].external_view = ibl_prefiltered_view;

    bg2_entries[1].binding = 1;
    bg2_entries[1].external_view = ibl_irradiance_view;

    bg2_entries[2].binding = 2;
    bg2_entries[2].external_view = ibl_brdf_lut_view;

    bg2_entries[3].binding = 3;
    bg2_entries[3].sampler = ready.ibl_sampler;

    rendering::BindGroupDesc bg2_desc;
    bg2_desc.layout = ready.ibl_bgl;
    bg2_desc.entries.assign(std::begin(bg2_entries), std::end(bg2_entries));
    auto bg2_handle = create_bind_group(fg, std::move(bg2_desc), "ibl_bg");

    // Skybox uniform buffer + bind group
    rendering::BufferDesc skybox_buf_desc;
    skybox_buf_desc.size = sizeof(SkyboxUniforms);
    skybox_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto skybox_uniform_buf_handle = create_buffer(fg, skybox_buf_desc, "skybox_uniforms");

    rendering::BindGroupEntry sky_bg_entries[3] = {};
    sky_bg_entries[0].binding = 0;
    sky_bg_entries[0].buffer = skybox_uniform_buf_handle;
    sky_bg_entries[0].buffer_size = sizeof(SkyboxUniforms);

    sky_bg_entries[1].binding = 1;
    sky_bg_entries[1].external_view = ibl_env_cubemap_view;

    sky_bg_entries[2].binding = 2;
    sky_bg_entries[2].sampler = ready.ibl_sampler;

    rendering::BindGroupDesc skybox_bg_desc;
    skybox_bg_desc.layout = ready.skybox_bgl;
    skybox_bg_desc.entries.assign(std::begin(sky_bg_entries), std::end(sky_bg_entries));
    auto skybox_bg_handle = create_bind_group(fg, std::move(skybox_bg_desc), "skybox_bg");

    // Capture values for the execute lambda
    auto queue = ctx.queue;
    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    auto elapsed_time = ctx.time;
    auto camera_pos = ctx.camera_position;
    auto* pipeline_handle = ready.pipeline.handle();
    auto skybox_pipeline_handle = ready.skybox_pipeline.handle();
    const auto& world = ctx.world;

    auto pass_builder = fg.add_pass("forward").color(color).read(shadow_out.shadow_array);
    for (uint32_t i = 0; i < eff_debug_count; ++i) {
        pass_builder.color(debug_handles[i]);
    }
    pass_builder.depth(depth).execute([=, &fg, &world](WGPURenderPassEncoder pass) {
        auto objs = world.get_objects();
        auto meshes = world.get_meshes();

        auto uniform_buf = fg.get_buffer_ref(uniform_buf_handle).handle();
        auto bg0 = fg.get_bind_group_ref(bg0_handle).handle();
        auto bg1 = fg.get_bind_group_ref(bg1_handle).handle();
        auto bg2 = fg.get_bind_group_ref(bg2_handle).handle();

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
        wgpuRenderPassEncoderSetBindGroup(pass, 1, bg1, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(pass, 2, bg2, 0, nullptr);

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
            auto skybox_bg = fg.get_bind_group_ref(skybox_bg_handle).handle();
            wgpuRenderPassEncoderSetPipeline(pass, skybox_pipeline_handle);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, skybox_bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        }
    });

    // Post-pass: SSAO
    if (auto* ssao = get_pass<rendering::SSAOPass>(); ssao && ssao->is_ready())
        ssao->add_to_frame_graph(fg, ctx, {gbuf_out.depth, gbuf_out.normals});

    return {color, depth};
}
