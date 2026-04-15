#include <core/diagnostics.h>
#include <core/rendering/halfFloat.h>
#include <core/rendering/iblResources.h>
#include <core/rendering/webgpu/device.h>
#include <core/rendering/webgpu/pipelineBuilder.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "embedded_ibl_shaders.h"

namespace pts::rendering {

namespace {

std::string load_shader(std::string_view key) {
    auto src = ibl_resources::get_resource(key);
    CHECK_MSG(src.has_value(), "IBL shader resource not found");
    // Slangc emits rgba32float + read_write for RWTexture<float4>, but WebGPU
    // requires rgba16float (filterable) and write (no read-write feature needed).
    // Patch the generated WGSL to match our RGBA16Float textures and WriteOnly BGLs.
    std::string wgsl(*src);
    for (std::string::size_type pos = 0;
         (pos = wgsl.find("rgba32float", pos)) != std::string::npos;)
        wgsl.replace(pos, 11, "rgba16float");
    for (std::string::size_type pos = 0; (pos = wgsl.find("read_write", pos)) != std::string::npos;)
        wgsl.replace(pos, 10, "write");
    return wgsl;
}

uint32_t div_ceil(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

// Create a cubemap texture with 6 array layers.
WGPUTexture create_cubemap_texture(WGPUDevice dev, uint32_t face_size, uint32_t mip_count,
                                   WGPUTextureFormat format, WGPUTextureUsage usage) {
    WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    desc.size = {face_size, face_size, 6};
    desc.dimension = WGPUTextureDimension_2D;
    desc.format = format;
    desc.usage = usage;
    desc.mipLevelCount = mip_count;
    desc.sampleCount = 1;
    auto tex = wgpuDeviceCreateTexture(dev, &desc);
    CHECK_MSG(tex, "Failed to create cubemap texture");
    return tex;
}

// Create a cube view of a 6-layer 2D texture.
WGPUTextureView create_cube_view(WGPUTexture tex, WGPUTextureFormat format, uint32_t mip_count) {
    WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    vd.format = format;
    vd.dimension = WGPUTextureViewDimension_Cube;
    vd.baseMipLevel = 0;
    vd.mipLevelCount = mip_count;
    vd.baseArrayLayer = 0;
    vd.arrayLayerCount = 6;
    auto view = wgpuTextureCreateView(tex, &vd);
    CHECK_MSG(view, "Failed to create cube texture view");
    return view;
}

// Create a 2DArray view of a single layer at a specific mip level.
WGPUTextureView create_single_layer_view(WGPUTexture tex, WGPUTextureFormat format, uint32_t mip,
                                         uint32_t layer) {
    WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    vd.format = format;
    vd.dimension = WGPUTextureViewDimension_2DArray;
    vd.baseMipLevel = mip;
    vd.mipLevelCount = 1;
    vd.baseArrayLayer = layer;
    vd.arrayLayerCount = 1;
    auto view = wgpuTextureCreateView(tex, &vd);
    CHECK_MSG(view, "Failed to create single layer texture view");
    return view;
}

// Create a 2DArray view of a specific mip level (all 6 layers).
WGPUTextureView create_mip_array_view(WGPUTexture tex, WGPUTextureFormat format, uint32_t mip) {
    WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    vd.format = format;
    vd.dimension = WGPUTextureViewDimension_2DArray;
    vd.baseMipLevel = mip;
    vd.mipLevelCount = 1;
    vd.baseArrayLayer = 0;
    vd.arrayLayerCount = 6;
    auto view = wgpuTextureCreateView(tex, &vd);
    CHECK_MSG(view, "Failed to create mip array texture view");
    return view;
}

WGPUTextureView create_2d_view(WGPUTexture tex, WGPUTextureFormat format) {
    WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    vd.format = format;
    vd.dimension = WGPUTextureViewDimension_2D;
    vd.mipLevelCount = 1;
    vd.arrayLayerCount = 1;
    auto view = wgpuTextureCreateView(tex, &vd);
    CHECK_MSG(view, "Failed to create 2D texture view");
    return view;
}

// The IBL compute shaders (.slang) declare RWTexture2D<float4> without a format
// annotation, so slang reflection yields `rgba32float`. At runtime we patch the
// generated WGSL to `rgba16float, write` (see load_shader above) and pair it
// with RGBA16Float textures. The BGLs below are open-coded to match that
// runtime format explicitly -- shader reflection can't tell us the target
// format. Keep these local to this translation unit.
WGPUBindGroupLayout create_brdf_lut_desc_layout(const webgpu::Device& device) {
    WGPUBindGroupLayoutEntry entries[2] = {};
    entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;

    entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
    entries[1].storageTexture.format = WGPUTextureFormat_RGBA16Float;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    desc.entryCount = 2;
    desc.entries = entries;
    return wgpuDeviceCreateBindGroupLayout(device.handle(), &desc);
}

WGPUBindGroupLayout create_equirect_desc_layout(const webgpu::Device& device) {
    WGPUBindGroupLayoutEntry entries[4] = {};
    entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;

    entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].texture.sampleType = WGPUTextureSampleType_Float;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

    entries[2] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Compute;
    entries[2].sampler.type = WGPUSamplerBindingType_Filtering;

    entries[3] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[3].binding = 3;
    entries[3].visibility = WGPUShaderStage_Compute;
    entries[3].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
    entries[3].storageTexture.format = WGPUTextureFormat_RGBA16Float;
    entries[3].storageTexture.viewDimension = WGPUTextureViewDimension_2DArray;

    WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    desc.entryCount = 4;
    desc.entries = entries;
    return wgpuDeviceCreateBindGroupLayout(device.handle(), &desc);
}

WGPUBindGroupLayout create_downsample_desc_layout(const webgpu::Device& device) {
    WGPUBindGroupLayoutEntry entries[3] = {};
    entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;

    entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].texture.sampleType = WGPUTextureSampleType_Float;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_2DArray;

    entries[2] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Compute;
    entries[2].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
    entries[2].storageTexture.format = WGPUTextureFormat_RGBA16Float;
    entries[2].storageTexture.viewDimension = WGPUTextureViewDimension_2DArray;

    WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    desc.entryCount = 3;
    desc.entries = entries;
    return wgpuDeviceCreateBindGroupLayout(device.handle(), &desc);
}

WGPUBindGroupLayout create_convolve_desc_layout(const webgpu::Device& device) {
    WGPUBindGroupLayoutEntry entries[4] = {};
    entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;

    entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].texture.sampleType = WGPUTextureSampleType_Float;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_Cube;

    entries[2] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Compute;
    entries[2].sampler.type = WGPUSamplerBindingType_Filtering;

    entries[3] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[3].binding = 3;
    entries[3].visibility = WGPUShaderStage_Compute;
    entries[3].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
    entries[3].storageTexture.format = WGPUTextureFormat_RGBA16Float;
    entries[3].storageTexture.viewDimension = WGPUTextureViewDimension_2DArray;

    WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    desc.entryCount = 4;
    desc.entries = entries;
    return wgpuDeviceCreateBindGroupLayout(device.handle(), &desc);
}

WGPUPipelineLayout make_pipeline_layout(WGPUDevice dev, WGPUBindGroupLayout desc_layout) {
    WGPUPipelineLayoutDescriptor desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    desc.bindGroupLayoutCount = 1;
    desc.bindGroupLayouts = &desc_layout;
    auto layout = wgpuDeviceCreatePipelineLayout(dev, &desc);
    CHECK_MSG(layout, "Failed to create pipeline layout");
    return layout;
}

}  // namespace

// ===========================================================================
// IblPipelines
// ===========================================================================

void IblPipelines::release() {
    if (m_brdf_lut_view) wgpuTextureViewRelease(m_brdf_lut_view);
    if (m_brdf_lut) wgpuTextureRelease(m_brdf_lut);
    // m_sampler is NOT released here -- it's owned by the FrameGraph sampler pool
    if (m_equirect_desc_layout) wgpuBindGroupLayoutRelease(m_equirect_desc_layout);
    if (m_downsample_desc_layout) wgpuBindGroupLayoutRelease(m_downsample_desc_layout);
    if (m_convolve_desc_layout) wgpuBindGroupLayoutRelease(m_convolve_desc_layout);
    if (m_brdf_lut_desc_layout) wgpuBindGroupLayoutRelease(m_brdf_lut_desc_layout);
    m_brdf_lut_view = nullptr;
    m_brdf_lut = nullptr;
    m_sampler = nullptr;
    m_equirect_desc_layout = nullptr;
    m_downsample_desc_layout = nullptr;
    m_convolve_desc_layout = nullptr;
    m_brdf_lut_desc_layout = nullptr;
    m_equirect_to_cube_pipeline.reset();
    m_downsample_pipeline.reset();
    m_irradiance_pipeline.reset();
    m_prefilter_pipeline.reset();
    m_brdf_lut_pipeline.reset();
    m_initialized = false;
}

IblPipelines::~IblPipelines() {
    release();
}

bool IblPipelines::is_ready() const noexcept {
    return m_initialized;
}

WGPUTextureView IblPipelines::brdf_lut_view() const noexcept {
    return m_brdf_lut_view;
}

WGPUSampler IblPipelines::sampler() const noexcept {
    return m_sampler;
}

WGPUComputePipeline IblPipelines::equirect_to_cube_pipeline() const noexcept {
    return m_equirect_to_cube_pipeline->handle();
}

WGPUComputePipeline IblPipelines::downsample_pipeline() const noexcept {
    return m_downsample_pipeline->handle();
}

WGPUComputePipeline IblPipelines::irradiance_pipeline() const noexcept {
    return m_irradiance_pipeline->handle();
}

WGPUComputePipeline IblPipelines::prefilter_pipeline() const noexcept {
    return m_prefilter_pipeline->handle();
}

WGPUBindGroupLayout IblPipelines::equirect_desc_layout() const noexcept {
    return m_equirect_desc_layout;
}

WGPUBindGroupLayout IblPipelines::downsample_desc_layout() const noexcept {
    return m_downsample_desc_layout;
}

WGPUBindGroupLayout IblPipelines::convolve_desc_layout() const noexcept {
    return m_convolve_desc_layout;
}

void IblPipelines::init(const webgpu::Device& device, WGPUQueue queue, WGPUSampler sampler) {
    PRECONDITION_MSG(!m_initialized, "IblPipelines already initialized");
    PRECONDITION(sampler != nullptr);
    auto dev = device.handle();

    // Bind group layouts
    m_brdf_lut_desc_layout = create_brdf_lut_desc_layout(device);
    m_equirect_desc_layout = create_equirect_desc_layout(device);
    m_downsample_desc_layout = create_downsample_desc_layout(device);
    m_convolve_desc_layout = create_convolve_desc_layout(device);

    // Pipelines
    {
        auto wgsl = load_shader("brdf_lut.wgsl");
        auto shader = device.create_shader_module_from_source(wgsl);
        auto layout = make_pipeline_layout(dev, m_brdf_lut_desc_layout);
        m_brdf_lut_pipeline = webgpu::ComputePipelineBuilder(device)
                                  .shader(shader)
                                  .entry_point("cs_main")
                                  .pipeline_layout(layout)
                                  .build();
        wgpuPipelineLayoutRelease(layout);
    }
    {
        auto wgsl = load_shader("equirect_to_cube.wgsl");
        auto shader = device.create_shader_module_from_source(wgsl);
        auto layout = make_pipeline_layout(dev, m_equirect_desc_layout);
        m_equirect_to_cube_pipeline = webgpu::ComputePipelineBuilder(device)
                                          .shader(shader)
                                          .entry_point("cs_main")
                                          .pipeline_layout(layout)
                                          .build();
        wgpuPipelineLayoutRelease(layout);
    }
    {
        auto wgsl = load_shader("downsample_cube.wgsl");
        auto shader = device.create_shader_module_from_source(wgsl);
        auto layout = make_pipeline_layout(dev, m_downsample_desc_layout);
        m_downsample_pipeline = webgpu::ComputePipelineBuilder(device)
                                    .shader(shader)
                                    .entry_point("cs_main")
                                    .pipeline_layout(layout)
                                    .build();
        wgpuPipelineLayoutRelease(layout);
    }
    {
        auto wgsl = load_shader("irradiance_convolve.wgsl");
        auto shader = device.create_shader_module_from_source(wgsl);
        auto layout = make_pipeline_layout(dev, m_convolve_desc_layout);
        m_irradiance_pipeline = webgpu::ComputePipelineBuilder(device)
                                    .shader(shader)
                                    .entry_point("cs_main")
                                    .pipeline_layout(layout)
                                    .build();
        wgpuPipelineLayoutRelease(layout);
    }
    {
        auto wgsl = load_shader("prefilter_env.wgsl");
        auto shader = device.create_shader_module_from_source(wgsl);
        auto layout = make_pipeline_layout(dev, m_convolve_desc_layout);
        m_prefilter_pipeline = webgpu::ComputePipelineBuilder(device)
                                   .shader(shader)
                                   .entry_point("cs_main")
                                   .pipeline_layout(layout)
                                   .build();
        wgpuPipelineLayoutRelease(layout);
    }

    // Sampler provided externally (shared via FrameGraph sampler pool)
    m_sampler = sampler;

    // Generate BRDF LUT
    generate_brdf_lut(device, queue);

    m_initialized = true;
}

void IblPipelines::generate_brdf_lut(const webgpu::Device& device, WGPUQueue queue) {
    auto dev = device.handle();
    constexpr uint32_t n = k_brdf_lut_size;

    {
        WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        desc.size = {n, n, 1};
        desc.format = WGPUTextureFormat_RGBA16Float;
        desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_StorageBinding |
                                                   WGPUTextureUsage_TextureBinding);
        desc.mipLevelCount = 1;
        desc.sampleCount = 1;
        desc.dimension = WGPUTextureDimension_2D;
        m_brdf_lut = wgpuDeviceCreateTexture(dev, &desc);
        CHECK_MSG(m_brdf_lut, "Failed to create BRDF LUT texture");
    }

    m_brdf_lut_view = create_2d_view(m_brdf_lut, WGPUTextureFormat_RGBA16Float);

    // Uniform buffer -- std140 pads to 16 bytes
    struct alignas(16) Params {
        uint32_t size;
    };
    Params params{n};
    auto uniform_buf = device.create_buffer(
        sizeof(params),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));
    wgpuQueueWriteBuffer(queue, uniform_buf.handle(), 0, &params, sizeof(params));

    // Storage texture view for writing
    auto storage_view = create_2d_view(m_brdf_lut, WGPUTextureFormat_RGBA16Float);

    // Bind group
    WGPUBindGroupEntry bg_entries[2] = {};
    bg_entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
    bg_entries[0].binding = 0;
    bg_entries[0].buffer = uniform_buf.handle();
    bg_entries[0].offset = 0;
    bg_entries[0].size = sizeof(params);

    bg_entries[1] = WGPU_BIND_GROUP_ENTRY_INIT;
    bg_entries[1].binding = 1;
    bg_entries[1].textureView = storage_view;

    WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg_desc.layout = m_brdf_lut_desc_layout;
    bg_desc.entryCount = 2;
    bg_desc.entries = bg_entries;
    auto bg = wgpuDeviceCreateBindGroup(dev, &bg_desc);
    CHECK_MSG(bg, "Failed to create BRDF LUT bind group");

    // Dispatch
    auto encoder = wgpuDeviceCreateCommandEncoder(dev, nullptr);
    auto pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, m_brdf_lut_pipeline->handle());
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, div_ceil(n, 8), div_ceil(n, 8), 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    auto cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bg);
    wgpuTextureViewRelease(storage_view);
}

// ===========================================================================
// IblResources
// ===========================================================================

void IblResources::release() {
    if (m_irradiance_view) wgpuTextureViewRelease(m_irradiance_view);
    if (m_irradiance) wgpuTextureRelease(m_irradiance);
    if (m_prefiltered_view) wgpuTextureViewRelease(m_prefiltered_view);
    if (m_prefiltered) wgpuTextureRelease(m_prefiltered);
    if (m_env_cube_view) wgpuTextureViewRelease(m_env_cube_view);
    if (m_env_cubemap) wgpuTextureRelease(m_env_cubemap);
    m_irradiance_view = nullptr;
    m_irradiance = nullptr;
    m_prefiltered_view = nullptr;
    m_prefiltered = nullptr;
    m_env_cube_view = nullptr;
    m_env_cubemap = nullptr;
    m_env_ready = false;
}

IblResources::~IblResources() {
    release();
}

IblResources::IblResources(IblResources&& o) noexcept
    : m_env_cubemap(std::exchange(o.m_env_cubemap, nullptr)),
      m_prefiltered(std::exchange(o.m_prefiltered, nullptr)),
      m_irradiance(std::exchange(o.m_irradiance, nullptr)),
      m_env_cube_view(std::exchange(o.m_env_cube_view, nullptr)),
      m_prefiltered_view(std::exchange(o.m_prefiltered_view, nullptr)),
      m_irradiance_view(std::exchange(o.m_irradiance_view, nullptr)),
      m_env_ready(std::exchange(o.m_env_ready, false)) {
}

IblResources& IblResources::operator=(IblResources&& o) noexcept {
    if (this != &o) {
        release();
        m_env_cubemap = std::exchange(o.m_env_cubemap, nullptr);
        m_prefiltered = std::exchange(o.m_prefiltered, nullptr);
        m_irradiance = std::exchange(o.m_irradiance, nullptr);
        m_env_cube_view = std::exchange(o.m_env_cube_view, nullptr);
        m_prefiltered_view = std::exchange(o.m_prefiltered_view, nullptr);
        m_irradiance_view = std::exchange(o.m_irradiance_view, nullptr);
        m_env_ready = std::exchange(o.m_env_ready, false);
    }
    return *this;
}

bool IblResources::is_ready() const noexcept {
    return m_env_ready;
}

WGPUTextureView IblResources::prefiltered_env_view() const noexcept {
    return m_prefiltered_view;
}

WGPUTextureView IblResources::env_cubemap_view() const noexcept {
    return m_env_cube_view;
}

WGPUTextureView IblResources::irradiance_view() const noexcept {
    return m_irradiance_view;
}

// ---------------------------------------------------------------------------
// set_environment -- full HDR equirect pipeline
// ---------------------------------------------------------------------------

void IblResources::set_environment(const IblPipelines& pipelines, const webgpu::Device& device,
                                   WGPUQueue queue, const float* hdr_rgba, uint32_t width,
                                   uint32_t height, UpAxis up_axis) {
    PRECONDITION(pipelines.is_ready());
    PRECONDITION(hdr_rgba != nullptr);
    PRECONDITION(width > 0 && height > 0);

    release();
    auto dev = device.handle();

    // Upload equirect as RGBA16Float 2D texture
    WGPUTexture equirect = nullptr;
    {
        WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        desc.size = {width, height, 1};
        desc.format = WGPUTextureFormat_RGBA16Float;
        desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding |
                                                   WGPUTextureUsage_CopyDst);
        desc.mipLevelCount = 1;
        desc.sampleCount = 1;
        desc.dimension = WGPUTextureDimension_2D;
        equirect = wgpuDeviceCreateTexture(dev, &desc);
        CHECK_MSG(equirect, "Failed to create equirect texture");

        std::vector<uint16_t> half_data(static_cast<size_t>(width) * height * 4);
        for (size_t i = 0; i < half_data.size(); ++i) {
            half_data[i] = float_to_half(hdr_rgba[i]);
        }

        WGPUTexelCopyBufferLayout layout = {};
        layout.offset = 0;
        layout.bytesPerRow = width * 4 * sizeof(uint16_t);
        layout.rowsPerImage = height;

        WGPUTexelCopyTextureInfo dest = {};
        dest.texture = equirect;
        dest.mipLevel = 0;
        dest.origin = {0, 0, 0};
        dest.aspect = WGPUTextureAspect_All;

        WGPUExtent3D extent = {width, height, 1};
        wgpuQueueWriteTexture(queue, &dest, half_data.data(), half_data.size() * sizeof(uint16_t),
                              &layout, &extent);
    }

    // Create env cubemap with full mip chain for mip-biased sampling.
    constexpr auto env_usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_StorageBinding |
                                      WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc);
    m_env_cubemap = create_cubemap_texture(dev, k_env_size, k_env_mip_count,
                                           WGPUTextureFormat_RGBA16Float, env_usage);

    // Prefiltered specular cubemap (separate from raw env).
    constexpr auto prefilter_usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_StorageBinding |
                                      WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst);
    m_prefiltered = create_cubemap_texture(dev, k_env_size, k_prefilter_mip_count,
                                           WGPUTextureFormat_RGBA16Float, prefilter_usage);

    constexpr auto irr_usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_StorageBinding |
                                                             WGPUTextureUsage_TextureBinding);
    m_irradiance =
        create_cubemap_texture(dev, k_irradiance_size, 1, WGPUTextureFormat_RGBA16Float, irr_usage);

    // Run compute passes
    convert_equirect_to_cubemap(pipelines, device, queue, equirect, up_axis);
    generate_env_mipmaps(pipelines, device, queue);

    // Copy mip 0 from env cubemap to prefiltered (roughness=0 = raw env)
    {
        auto encoder = wgpuDeviceCreateCommandEncoder(dev, nullptr);

        WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        src.texture = m_env_cubemap;
        src.mipLevel = 0;

        WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        dst.texture = m_prefiltered;
        dst.mipLevel = 0;

        WGPUExtent3D extent = {k_env_size, k_env_size, 6};
        wgpuCommandEncoderCopyTextureToTexture(encoder, &src, &dst, &extent);

        auto cmd = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);
    }

    convolve_irradiance(pipelines, device, queue);
    prefilter_specular(pipelines, device, queue);

    // Create final views
    m_env_cube_view =
        create_cube_view(m_env_cubemap, WGPUTextureFormat_RGBA16Float, k_env_mip_count);
    m_prefiltered_view =
        create_cube_view(m_prefiltered, WGPUTextureFormat_RGBA16Float, k_prefilter_mip_count);
    m_irradiance_view = create_cube_view(m_irradiance, WGPUTextureFormat_RGBA16Float, 1);

    wgpuTextureRelease(equirect);
    m_env_ready = true;
}

// ---------------------------------------------------------------------------
// set_uniform_environment -- solid color 1x1 cubemaps
// ---------------------------------------------------------------------------

void IblResources::set_uniform_environment(const webgpu::Device& device, WGPUQueue queue, float r,
                                           float g, float b) {
    release();
    auto dev = device.handle();

    uint16_t hr = float_to_half(r);
    uint16_t hg = float_to_half(g);
    uint16_t hb = float_to_half(b);
    uint16_t ha = float_to_half(1.0f);

    auto fill_cubemap = [&](WGPUTexture tex, uint32_t face_size, uint32_t mip_count) {
        for (uint32_t mip = 0; mip < mip_count; ++mip) {
            uint32_t mip_size = std::max(face_size >> mip, 1u);
            std::vector<uint16_t> face_data(static_cast<size_t>(mip_size) * mip_size * 4);
            for (size_t px = 0; px < static_cast<size_t>(mip_size) * mip_size; ++px) {
                face_data[px * 4 + 0] = hr;
                face_data[px * 4 + 1] = hg;
                face_data[px * 4 + 2] = hb;
                face_data[px * 4 + 3] = ha;
            }

            for (uint32_t face = 0; face < 6; ++face) {
                WGPUTexelCopyBufferLayout layout = {};
                layout.offset = 0;
                layout.bytesPerRow = mip_size * 4 * sizeof(uint16_t);
                layout.rowsPerImage = mip_size;

                WGPUTexelCopyTextureInfo dest = {};
                dest.texture = tex;
                dest.mipLevel = mip;
                dest.origin = {0, 0, face};
                dest.aspect = WGPUTextureAspect_All;

                WGPUExtent3D extent = {mip_size, mip_size, 1};
                wgpuQueueWriteTexture(queue, &dest, face_data.data(),
                                      face_data.size() * sizeof(uint16_t), &layout, &extent);
            }
        }
    };

    // For uniform color, use CopyDst instead of StorageBinding (no compute needed)
    constexpr auto tex_usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst);

    m_env_cubemap = create_cubemap_texture(dev, 1, 1, WGPUTextureFormat_RGBA16Float, tex_usage);
    m_prefiltered = create_cubemap_texture(dev, 1, 1, WGPUTextureFormat_RGBA16Float, tex_usage);
    m_irradiance = create_cubemap_texture(dev, 1, 1, WGPUTextureFormat_RGBA16Float, tex_usage);

    fill_cubemap(m_env_cubemap, 1, 1);
    fill_cubemap(m_prefiltered, 1, 1);

    // Irradiance of a uniform environment = PI * color
    uint16_t ir = float_to_half(r * 3.14159265f);
    uint16_t ig = float_to_half(g * 3.14159265f);
    uint16_t ib = float_to_half(b * 3.14159265f);
    uint16_t irr_pixel[4] = {ir, ig, ib, ha};
    for (uint32_t face = 0; face < 6; ++face) {
        WGPUTexelCopyBufferLayout layout = {};
        layout.offset = 0;
        layout.bytesPerRow = 4 * sizeof(uint16_t);
        layout.rowsPerImage = 1;

        WGPUTexelCopyTextureInfo dest = {};
        dest.texture = m_irradiance;
        dest.mipLevel = 0;
        dest.origin = {0, 0, face};
        dest.aspect = WGPUTextureAspect_All;

        WGPUExtent3D extent = {1, 1, 1};
        wgpuQueueWriteTexture(queue, &dest, irr_pixel, sizeof(irr_pixel), &layout, &extent);
    }

    m_env_cube_view = create_cube_view(m_env_cubemap, WGPUTextureFormat_RGBA16Float, 1);
    m_prefiltered_view = create_cube_view(m_prefiltered, WGPUTextureFormat_RGBA16Float, 1);
    m_irradiance_view = create_cube_view(m_irradiance, WGPUTextureFormat_RGBA16Float, 1);
    m_env_ready = true;
}

// ---------------------------------------------------------------------------
// Equirect -> Cubemap
// ---------------------------------------------------------------------------

void IblResources::convert_equirect_to_cubemap(const IblPipelines& pipelines,
                                               const webgpu::Device& device, WGPUQueue queue,
                                               WGPUTexture equirect, UpAxis up_axis) {
    auto dev = device.handle();

    // Dispatch one face at a time with a single-layer output view.
    // Writing to multiple array layers via textureStore in a single dispatch
    // silently drops writes to layers > 0 on some D3D12 backends (Dawn/Tint
    // WGSL->HLSL codegen issue with mixed u32/i32 textureStore coordinates).
    struct alignas(16) Params {
        uint32_t size;
        uint32_t up_axis;
        uint32_t face;
        uint32_t _pad;
    };

    auto uniform_buf = device.create_buffer(
        sizeof(Params),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));
    auto equirect_view = create_2d_view(equirect, WGPUTextureFormat_RGBA16Float);

    for (uint32_t face = 0; face < 6; ++face) {
        Params params{k_env_size, static_cast<uint32_t>(up_axis), face, 0};
        wgpuQueueWriteBuffer(queue, uniform_buf.handle(), 0, &params, sizeof(params));

        auto output_view =
            create_single_layer_view(m_env_cubemap, WGPUTextureFormat_RGBA16Float, 0, face);

        WGPUBindGroupEntry entries[4] = {};
        entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[0].binding = 0;
        entries[0].buffer = uniform_buf.handle();
        entries[0].offset = 0;
        entries[0].size = sizeof(Params);

        entries[1] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[1].binding = 1;
        entries[1].textureView = equirect_view;

        entries[2] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[2].binding = 2;
        entries[2].sampler = pipelines.sampler();

        entries[3] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[3].binding = 3;
        entries[3].textureView = output_view;

        WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bg_desc.layout = pipelines.equirect_desc_layout();
        bg_desc.entryCount = 4;
        bg_desc.entries = entries;
        auto bg = wgpuDeviceCreateBindGroup(dev, &bg_desc);
        CHECK_MSG(bg, "Failed to create equirect bind group");

        auto encoder = wgpuDeviceCreateCommandEncoder(dev, nullptr);
        auto pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, pipelines.equirect_to_cube_pipeline());
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, div_ceil(k_env_size, 8),
                                                 div_ceil(k_env_size, 8), 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        auto cmd = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);
        wgpuBindGroupRelease(bg);
        wgpuTextureViewRelease(output_view);
    }

    wgpuTextureViewRelease(equirect_view);
}

// ---------------------------------------------------------------------------
// Env cubemap mipmap generation (box-filter downsample)
// ---------------------------------------------------------------------------

void IblResources::generate_env_mipmaps(const IblPipelines& pipelines, const webgpu::Device& device,
                                        WGPUQueue queue) {
    auto dev = device.handle();

    struct alignas(16) Params {
        uint32_t output_size;
        uint32_t face;
        uint32_t _pad[2];
    };

    auto uniform_buf = device.create_buffer(
        sizeof(Params),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    for (uint32_t mip = 1; mip < k_env_mip_count; ++mip) {
        uint32_t output_size = k_env_size >> mip;

        for (uint32_t face = 0; face < 6; ++face) {
            Params params{output_size, face, {0, 0}};
            wgpuQueueWriteBuffer(queue, uniform_buf.handle(), 0, &params, sizeof(params));

            auto input_view =
                create_mip_array_view(m_env_cubemap, WGPUTextureFormat_RGBA16Float, mip - 1);
            auto output_view =
                create_single_layer_view(m_env_cubemap, WGPUTextureFormat_RGBA16Float, mip, face);

            WGPUBindGroupEntry entries[3] = {};
            entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
            entries[0].binding = 0;
            entries[0].buffer = uniform_buf.handle();
            entries[0].offset = 0;
            entries[0].size = sizeof(Params);

            entries[1] = WGPU_BIND_GROUP_ENTRY_INIT;
            entries[1].binding = 1;
            entries[1].textureView = input_view;

            entries[2] = WGPU_BIND_GROUP_ENTRY_INIT;
            entries[2].binding = 2;
            entries[2].textureView = output_view;

            WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
            bg_desc.layout = pipelines.downsample_desc_layout();
            bg_desc.entryCount = 3;
            bg_desc.entries = entries;
            auto bg = wgpuDeviceCreateBindGroup(dev, &bg_desc);
            CHECK_MSG(bg, "Failed to create downsample bind group");

            auto encoder = wgpuDeviceCreateCommandEncoder(dev, nullptr);
            auto pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, pipelines.downsample_pipeline());
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, div_ceil(output_size, 8),
                                                     div_ceil(output_size, 8), 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);

            auto cmd = wgpuCommandEncoderFinish(encoder, nullptr);
            wgpuQueueSubmit(queue, 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(encoder);
            wgpuBindGroupRelease(bg);
            wgpuTextureViewRelease(output_view);
            wgpuTextureViewRelease(input_view);
        }
    }
}

// ---------------------------------------------------------------------------
// Irradiance convolution
// ---------------------------------------------------------------------------

void IblResources::convolve_irradiance(const IblPipelines& pipelines, const webgpu::Device& device,
                                       WGPUQueue queue) {
    auto dev = device.handle();

    struct alignas(16) Params {
        uint32_t size;
        uint32_t env_size;
        uint32_t face;
        uint32_t _pad;
    };

    auto uniform_buf = device.create_buffer(
        sizeof(Params),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    auto input_view =
        create_cube_view(m_env_cubemap, WGPUTextureFormat_RGBA16Float, k_env_mip_count);

    for (uint32_t face = 0; face < 6; ++face) {
        Params params{k_irradiance_size, k_env_size, face, 0};
        wgpuQueueWriteBuffer(queue, uniform_buf.handle(), 0, &params, sizeof(params));

        auto output_view =
            create_single_layer_view(m_irradiance, WGPUTextureFormat_RGBA16Float, 0, face);

        WGPUBindGroupEntry entries[4] = {};
        entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[0].binding = 0;
        entries[0].buffer = uniform_buf.handle();
        entries[0].offset = 0;
        entries[0].size = sizeof(Params);

        entries[1] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[1].binding = 1;
        entries[1].textureView = input_view;

        entries[2] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[2].binding = 2;
        entries[2].sampler = pipelines.sampler();

        entries[3] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[3].binding = 3;
        entries[3].textureView = output_view;

        WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bg_desc.layout = pipelines.convolve_desc_layout();
        bg_desc.entryCount = 4;
        bg_desc.entries = entries;
        auto bg = wgpuDeviceCreateBindGroup(dev, &bg_desc);
        CHECK_MSG(bg, "Failed to create irradiance bind group");

        auto encoder = wgpuDeviceCreateCommandEncoder(dev, nullptr);
        auto pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, pipelines.irradiance_pipeline());
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, div_ceil(k_irradiance_size, 8),
                                                 div_ceil(k_irradiance_size, 8), 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        auto cmd = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);
        wgpuBindGroupRelease(bg);
        wgpuTextureViewRelease(output_view);
    }

    wgpuTextureViewRelease(input_view);
}

// ---------------------------------------------------------------------------
// Specular prefilter (one dispatch per mip level, mips 1 .. k_prefilter_mip_count-1)
// ---------------------------------------------------------------------------

void IblResources::prefilter_specular(const IblPipelines& pipelines, const webgpu::Device& device,
                                      WGPUQueue queue) {
    auto dev = device.handle();

    // Read from env cubemap with full mip chain for mip-biased sampling
    auto input_view =
        create_cube_view(m_env_cubemap, WGPUTextureFormat_RGBA16Float, k_env_mip_count);

    struct alignas(16) Params {
        uint32_t size;
        float roughness;
        uint32_t env_size;
        uint32_t face;
    };

    auto uniform_buf = device.create_buffer(
        sizeof(Params),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    for (uint32_t mip = 1; mip < k_prefilter_mip_count; ++mip) {
        uint32_t mip_size = k_env_size >> mip;
        float roughness = static_cast<float>(mip) / static_cast<float>(k_prefilter_mip_count - 1);

        for (uint32_t face = 0; face < 6; ++face) {
            Params params{mip_size, roughness, k_env_size, face};
            wgpuQueueWriteBuffer(queue, uniform_buf.handle(), 0, &params, sizeof(params));

            auto output_view =
                create_single_layer_view(m_prefiltered, WGPUTextureFormat_RGBA16Float, mip, face);

            WGPUBindGroupEntry entries[4] = {};
            entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
            entries[0].binding = 0;
            entries[0].buffer = uniform_buf.handle();
            entries[0].offset = 0;
            entries[0].size = sizeof(Params);

            entries[1] = WGPU_BIND_GROUP_ENTRY_INIT;
            entries[1].binding = 1;
            entries[1].textureView = input_view;

            entries[2] = WGPU_BIND_GROUP_ENTRY_INIT;
            entries[2].binding = 2;
            entries[2].sampler = pipelines.sampler();

            entries[3] = WGPU_BIND_GROUP_ENTRY_INIT;
            entries[3].binding = 3;
            entries[3].textureView = output_view;

            WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
            bg_desc.layout = pipelines.convolve_desc_layout();
            bg_desc.entryCount = 4;
            bg_desc.entries = entries;
            auto bg = wgpuDeviceCreateBindGroup(dev, &bg_desc);
            CHECK_MSG(bg, "Failed to create prefilter bind group");

            auto encoder = wgpuDeviceCreateCommandEncoder(dev, nullptr);
            auto pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, pipelines.prefilter_pipeline());
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, div_ceil(mip_size, 8),
                                                     div_ceil(mip_size, 8), 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);

            auto cmd = wgpuCommandEncoderFinish(encoder, nullptr);
            wgpuQueueSubmit(queue, 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(encoder);
            wgpuBindGroupRelease(bg);
            wgpuTextureViewRelease(output_view);
        }
    }

    wgpuTextureViewRelease(input_view);
}

}  // namespace pts::rendering
