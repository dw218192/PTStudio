#include "forwardPass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/camera.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/iblResources.h>
#include <core/rendering/passContext.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/rendererRegistry.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <renderers/forward/generated/shader_metadata.h>

#include <glm/glm.hpp>

using namespace pts;
using namespace pts::editor;

REGISTER_RENDERER("Forward", ForwardPass);

ForwardPass::ForwardPass(const rendering::ShaderLoader& sl) : IRenderer(sl) {
    add_pass<rendering::ShadowMapPass>(sl);
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

static WGPUBindGroup create_bind_group(WGPUDevice device, WGPUBindGroupLayout layout,
                                       WGPUBuffer uniform_buf, WGPUBuffer material_buf,
                                       std::size_t material_buf_size, WGPUBuffer light_buf,
                                       std::size_t light_buf_size, WGPUTextureView ltc_mat_view,
                                       WGPUTextureView ltc_amp_view, WGPUSampler ltc_sampler,
                                       WGPUTextureView scene_tex_view,
                                       WGPUSampler scene_tex_sampler) {
    WGPUBindGroupEntry entries[8] = {};

    entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
    entries[0].binding = 0;
    entries[0].buffer = uniform_buf;
    entries[0].offset = 0;
    entries[0].size = sizeof(ForwardUniforms);

    entries[1] = WGPU_BIND_GROUP_ENTRY_INIT;
    entries[1].binding = 1;
    entries[1].buffer = material_buf;
    entries[1].offset = 0;
    entries[1].size = material_buf_size;

    entries[2] = WGPU_BIND_GROUP_ENTRY_INIT;
    entries[2].binding = 2;
    entries[2].buffer = light_buf;
    entries[2].offset = 0;
    entries[2].size = light_buf_size;

    entries[3] = WGPU_BIND_GROUP_ENTRY_INIT;
    entries[3].binding = 3;
    entries[3].textureView = ltc_mat_view;

    entries[4] = WGPU_BIND_GROUP_ENTRY_INIT;
    entries[4].binding = 4;
    entries[4].textureView = ltc_amp_view;

    entries[5] = WGPU_BIND_GROUP_ENTRY_INIT;
    entries[5].binding = 5;
    entries[5].sampler = ltc_sampler;

    entries[6] = WGPU_BIND_GROUP_ENTRY_INIT;
    entries[6].binding = 6;
    entries[6].textureView = scene_tex_view;

    entries[7] = WGPU_BIND_GROUP_ENTRY_INIT;
    entries[7].binding = 7;
    entries[7].sampler = scene_tex_sampler;

    WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg_desc.layout = layout;
    bg_desc.entryCount = 8;
    bg_desc.entries = entries;
    return wgpuDeviceCreateBindGroup(device, &bg_desc);
}

ForwardPass::~ForwardPass() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group) wgpuBindGroupRelease(ready->bind_group);
        if (ready->bind_group_layout) wgpuBindGroupLayoutRelease(ready->bind_group_layout);
        if (ready->shadow_recv_bgl) wgpuBindGroupLayoutRelease(ready->shadow_recv_bgl);
        if (ready->shadow_sampler) wgpuSamplerRelease(ready->shadow_sampler);
        if (ready->ibl_bgl) wgpuBindGroupLayoutRelease(ready->ibl_bgl);
        if (ready->ibl_sampler) wgpuSamplerRelease(ready->ibl_sampler);
        if (ready->fallback_cube_view) wgpuTextureViewRelease(ready->fallback_cube_view);
        if (ready->fallback_cube_tex) wgpuTextureRelease(ready->fallback_cube_tex);
        if (ready->fallback_2d_view) wgpuTextureViewRelease(ready->fallback_2d_view);
        if (ready->fallback_2d_tex) wgpuTextureRelease(ready->fallback_2d_tex);
    }
}

static constexpr const char* k_debug_target_names[] = {
    "Normals", "Base Color", "Direct Diffuse", "Direct Specular", "IBL Diffuse", "IBL Specular",
};
static constexpr uint32_t k_debug_target_count =
    static_cast<uint32_t>(sizeof(k_debug_target_names) / sizeof(k_debug_target_names[0]));

auto ForwardPass::name() const noexcept -> std::string_view {
    return "forward";
}

auto ForwardPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

auto ForwardPass::debug_target_names() const noexcept -> std::pair<const char* const*, uint32_t> {
    return {k_debug_target_names, k_debug_target_count};
}

void ForwardPass::do_renderer_setup(const webgpu::Device& device) {
    // Release existing state for re-entry (hot-reload)
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group) wgpuBindGroupRelease(ready->bind_group);
        if (ready->bind_group_layout) wgpuBindGroupLayoutRelease(ready->bind_group_layout);
        if (ready->shadow_recv_bgl) wgpuBindGroupLayoutRelease(ready->shadow_recv_bgl);
        if (ready->shadow_sampler) wgpuSamplerRelease(ready->shadow_sampler);
        if (ready->ibl_bgl) wgpuBindGroupLayoutRelease(ready->ibl_bgl);
        if (ready->ibl_sampler) wgpuSamplerRelease(ready->ibl_sampler);
        if (ready->fallback_cube_view) wgpuTextureViewRelease(ready->fallback_cube_view);
        if (ready->fallback_cube_tex) wgpuTextureRelease(ready->fallback_cube_tex);
        if (ready->fallback_2d_view) wgpuTextureViewRelease(ready->fallback_2d_view);
        if (ready->fallback_2d_tex) wgpuTextureRelease(ready->fallback_2d_tex);
    }

    auto [dbg_names_setup, dbg_count_setup] = effective_debug_target_names();
    auto shader_src = load_pass_shader("renderers/forward/generated/shaders/forward.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    uint32_t initial_capacity = 64;
    auto uniform_buffer = device.create_buffer(
        k_uniform_align * initial_capacity,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

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
                       .depth_format(WGPUTextureFormat_Depth24Plus)
                       .depth_write(true)
                       .depth_compare(WGPUCompareFunction_Less)
                       .cull_mode(WGPUCullMode_Back)
                       .pipeline_layout(pipeline_layout)
                       .vertex_layout<forward_shader::VertexLayout>();
    for (uint32_t i = 0; i < dbg_count_setup; ++i) {
        builder.color_format(WGPUTextureFormat_RGBA8Unorm, i + 1);
    }
    auto pipeline = builder.build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    rendering::LtcTextures ltc;
    ltc.init(device);

    m_state = Ready{
        std::move(shader),
        std::move(pipeline),
        std::move(uniform_buffer),
        nullptr,
        bind_group_layout,
        initial_capacity,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        std::move(ltc),
        shadow_recv_bgl,
        shadow_sampler,
        ibl_bgl,
        ibl_sampler,
        fallback_cube_tex,
        fallback_cube_view,
        fallback_2d_tex,
        fallback_2d_view,
    };
}

bool ForwardPass::ensure_capacity(const webgpu::Device& device, uint32_t object_count) {
    auto& ready = std::get<Ready>(m_state);
    if (object_count <= ready.capacity) return false;

    uint32_t new_capacity = ready.capacity;
    while (new_capacity < object_count) {
        new_capacity *= 2;
    }

    ready.uniform_buffer = device.create_buffer(
        k_uniform_align * new_capacity,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    ready.capacity = new_capacity;
    return true;
}

void ForwardPass::do_add_to_frame_graph(rendering::FrameGraph& fg,
                                        const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());

    auto& ready = std::get<Ready>(m_state);

    auto objects = ctx.world.get_objects();
    auto object_count = static_cast<uint32_t>(objects.size());
    bool bind_group_dirty = false;
    if (object_count > 0) {
        bind_group_dirty = ensure_capacity(ctx.device, object_count);
    }

    // Detect RenderWorld buffer reallocation and rebuild bind group
    auto& light_buf = ctx.world.light_buffer();
    auto& mat_buf = ctx.world.material_buffer();
    auto light_count = ctx.world.gpu_light_count();

    auto scene_tex_view = ctx.world.texture_array_view();
    auto scene_tex_sampler = ctx.world.texture_sampler();

    if (ready.bind_group == nullptr || bind_group_dirty ||
        light_buf.handle() != ready.cached_light_buf ||
        mat_buf.handle() != ready.cached_material_buf ||
        scene_tex_view != ready.cached_texture_view ||
        scene_tex_sampler != ready.cached_texture_sampler) {
        if (ready.bind_group) {
            wgpuBindGroupRelease(ready.bind_group);
        }
        ready.bind_group = create_bind_group(
            ctx.device.handle(), ready.bind_group_layout, ready.uniform_buffer.handle(),
            mat_buf.handle(), mat_buf.size(), light_buf.handle(), light_buf.size(),
            ready.ltc_textures.mat_view(), ready.ltc_textures.amp_view(),
            ready.ltc_textures.sampler(), scene_tex_view, scene_tex_sampler);
        ready.cached_light_buf = light_buf.handle();
        ready.cached_material_buf = mat_buf.handle();
        ready.cached_texture_view = scene_tex_view;
        ready.cached_texture_sampler = scene_tex_sampler;
    }

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

    auto [eff_debug_names, eff_debug_count] = effective_debug_target_names();

    rendering::TextureDesc debug_desc;
    debug_desc.width = ctx.viewport_width;
    debug_desc.height = ctx.viewport_height;
    debug_desc.format = WGPUTextureFormat_RGBA8Unorm;
    debug_desc.clear_color = {0, 0, 0, 1};

    rendering::ResourceHandle debug_handles[k_debug_target_count];
    for (uint32_t i = 0; i < eff_debug_count; ++i) {
        debug_handles[i] =
            fg.find_or_create(std::string("debug_") + eff_debug_names[i], debug_desc);
    }

    auto queue = ctx.queue;
    auto view_mat = ctx.view_matrix;
    auto proj_mat = ctx.proj_matrix;
    auto elapsed_time = ctx.time;
    auto camera_pos = ctx.camera_position;
    auto* pipeline_handle = ready.pipeline.handle();
    auto uniform_buf = ready.uniform_buffer.handle();
    auto bind_group = ready.bind_group;
    const auto& world = ctx.world;

    // Capture shadow resources for the execute lambda
    auto dev = ctx.device.handle();
    auto shadow_recv_bgl = ready.shadow_recv_bgl;
    auto shadow_samp = ready.shadow_sampler;
    auto shadow_array_view = get_pass<rendering::ShadowMapPass>()->shadow_array_view();
    auto& sd = rendering::ShadowPassData::get_or_create(ctx.world);
    PRECONDITION(sd.info_buffer.is_valid());
    auto shadow_info_handle = sd.info_buffer.handle();
    auto shadow_info_size = sd.info_buffer.size();

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
    auto ibl_ready = ibl.is_ready();

    {
        PTS_ZONE_NAMED("forward uniform upload");
        for (uint32_t i = 0; i < object_count; ++i) {
            if (!objects[i].active()) continue;
            const auto& obj = objects[i];
            ForwardUniforms u{};
            u.mvp = proj_mat * view_mat * obj->transform;
            u.model = obj->transform;
            u.camera_pos = camera_pos;
            u.time = elapsed_time;
            u.material_index = obj->material_index;
            u.light_count = light_count;
            u.ibl_dome_modulation = ibl_ready ? dome_mod : glm::vec3{0.0f};
            u.ibl_mip_count = rendering::IblResources::k_prefilter_mip_count;
            wgpuQueueWriteBuffer(queue, uniform_buf, i * k_uniform_align, &u, sizeof(u));
        }
    }

    // Capture IBL bind group resources (use fallback textures when IBL not ready)
    auto ibl_bgl = ready.ibl_bgl;
    auto ibl_samp = ready.ibl_sampler;
    auto ibl_prefiltered_view = ibl_ready ? ibl.prefiltered_env_view() : ready.fallback_cube_view;
    auto ibl_irradiance_view = ibl_ready ? ibl.irradiance_view() : ready.fallback_cube_view;
    auto ibl_brdf_lut_view = ibl_ready ? ibl.brdf_lut_view() : ready.fallback_2d_view;

    auto pass_builder = fg.add_pass("forward").color(color);
    for (uint32_t i = 0; i < eff_debug_count; ++i) {
        pass_builder.color(debug_handles[i]);
    }
    pass_builder.depth(depth).execute([=, &world](WGPURenderPassEncoder pass) {
        auto objs = world.get_objects();
        auto meshes = world.get_meshes();

        // Shadow bind group (group 1) — created per-frame
        WGPUBindGroupEntry shadow_bg_entries[3] = {};
        shadow_bg_entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
        shadow_bg_entries[0].binding = 0;
        shadow_bg_entries[0].buffer = shadow_info_handle;
        shadow_bg_entries[0].size = shadow_info_size;

        shadow_bg_entries[1] = WGPU_BIND_GROUP_ENTRY_INIT;
        shadow_bg_entries[1].binding = 1;
        shadow_bg_entries[1].textureView = shadow_array_view;

        shadow_bg_entries[2] = WGPU_BIND_GROUP_ENTRY_INIT;
        shadow_bg_entries[2].binding = 2;
        shadow_bg_entries[2].sampler = shadow_samp;

        WGPUBindGroupDescriptor shadow_bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        shadow_bg_desc.layout = shadow_recv_bgl;
        shadow_bg_desc.entryCount = 3;
        shadow_bg_desc.entries = shadow_bg_entries;
        auto shadow_bg = wgpuDeviceCreateBindGroup(dev, &shadow_bg_desc);

        // IBL bind group (group 2) — created per-frame
        WGPUBindGroupEntry ibl_bg_entries[4] = {};
        ibl_bg_entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
        ibl_bg_entries[0].binding = 0;
        ibl_bg_entries[0].textureView = ibl_prefiltered_view;

        ibl_bg_entries[1] = WGPU_BIND_GROUP_ENTRY_INIT;
        ibl_bg_entries[1].binding = 1;
        ibl_bg_entries[1].textureView = ibl_irradiance_view;

        ibl_bg_entries[2] = WGPU_BIND_GROUP_ENTRY_INIT;
        ibl_bg_entries[2].binding = 2;
        ibl_bg_entries[2].textureView = ibl_brdf_lut_view;

        ibl_bg_entries[3] = WGPU_BIND_GROUP_ENTRY_INIT;
        ibl_bg_entries[3].binding = 3;
        ibl_bg_entries[3].sampler = ibl_samp;

        WGPUBindGroupDescriptor ibl_bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        ibl_bg_desc.layout = ibl_bgl;
        ibl_bg_desc.entryCount = 4;
        ibl_bg_desc.entries = ibl_bg_entries;
        auto ibl_bg = wgpuDeviceCreateBindGroup(dev, &ibl_bg_desc);

        wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
        wgpuRenderPassEncoderSetBindGroup(pass, 1, shadow_bg, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(pass, 2, ibl_bg, 0, nullptr);

        for (uint32_t i = 0; i < static_cast<uint32_t>(objs.size()); ++i) {
            if (!objs[i].active()) continue;
            uint32_t dyn_offset = i * k_uniform_align;
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 1, &dyn_offset);
            const auto& mesh = meshes[objs[i]->mesh_index];
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh->vertex_buffer.handle(), 0,
                                                 mesh->vertex_buffer.size());
            wgpuRenderPassEncoderSetIndexBuffer(pass, mesh->index_buffer.handle(),
                                                WGPUIndexFormat_Uint32, 0,
                                                mesh->index_buffer.size());
            wgpuRenderPassEncoderDrawIndexed(pass, mesh->index_count, 1, 0, 0, 0);
        }

        wgpuBindGroupRelease(shadow_bg);
        wgpuBindGroupRelease(ibl_bg);
    });
}
