#pragma once

#include <core/rendering/upAxis.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <optional>

namespace pts::webgpu {
class Device;
}

namespace pts::rendering {

// IBL constants — shared between IblPipelines and IblResources.
static constexpr uint32_t k_env_size = 256;
static constexpr uint32_t k_irradiance_size = 64;
static constexpr uint32_t k_brdf_lut_size = 512;
static constexpr uint32_t k_prefilter_mip_count = 6;
static constexpr uint32_t k_env_mip_count = 9;  // log2(k_env_size) + 1

/// Init-once compute infrastructure for IBL: pipelines, bind group
/// layouts, sampler, and BRDF LUT. Non-copyable, non-movable.
class IblPipelines {
   public:
    IblPipelines() = default;
    ~IblPipelines();

    IblPipelines(const IblPipelines&) = delete;
    IblPipelines& operator=(const IblPipelines&) = delete;
    IblPipelines(IblPipelines&&) = delete;
    IblPipelines& operator=(IblPipelines&&) = delete;

    /// Create compute pipelines and generate the BRDF LUT.
    /// The sampler is provided externally (e.g. from FrameGraph::sampler()).
    void init(const webgpu::Device& device, WGPUQueue queue, WGPUSampler sampler);

    bool is_ready() const noexcept;

    WGPUTextureView brdf_lut_view() const noexcept;
    WGPUSampler sampler() const noexcept;

    // Pipeline handle accessors (used by IblResources compute passes).
    WGPUComputePipeline equirect_to_cube_pipeline() const noexcept;
    WGPUComputePipeline downsample_pipeline() const noexcept;
    WGPUComputePipeline irradiance_pipeline() const noexcept;
    WGPUComputePipeline prefilter_pipeline() const noexcept;

    // Descriptor layout accessors.
    WGPUBindGroupLayout equirect_desc_layout() const noexcept;
    WGPUBindGroupLayout downsample_desc_layout() const noexcept;
    WGPUBindGroupLayout convolve_desc_layout() const noexcept;

   private:
    void release();
    void generate_brdf_lut(const webgpu::Device& device, WGPUQueue queue);

    std::optional<webgpu::ComputePipeline> m_equirect_to_cube_pipeline;
    std::optional<webgpu::ComputePipeline> m_downsample_pipeline;
    std::optional<webgpu::ComputePipeline> m_irradiance_pipeline;
    std::optional<webgpu::ComputePipeline> m_prefilter_pipeline;
    std::optional<webgpu::ComputePipeline> m_brdf_lut_pipeline;

    WGPUBindGroupLayout m_equirect_desc_layout = nullptr;
    WGPUBindGroupLayout m_downsample_desc_layout = nullptr;
    WGPUBindGroupLayout m_convolve_desc_layout = nullptr;
    WGPUBindGroupLayout m_brdf_lut_desc_layout = nullptr;

    WGPUSampler m_sampler = nullptr;
    WGPUTexture m_brdf_lut = nullptr;
    WGPUTextureView m_brdf_lut_view = nullptr;

    bool m_initialized = false;
};

/// Per-environment IBL textures: cubemap, prefiltered specular, and
/// irradiance map. Uses IblPipelines for GPU compute operations.
class IblResources {
   public:
    IblResources() = default;
    ~IblResources();

    IblResources(const IblResources&) = delete;
    IblResources& operator=(const IblResources&) = delete;
    IblResources(IblResources&&) noexcept;
    IblResources& operator=(IblResources&&) noexcept;

    /// Set environment from equirectangular HDR pixel data (RGBA float32).
    /// Triggers equirect->cube, irradiance convolution, specular prefilter.
    void set_environment(const IblPipelines& pipelines, const webgpu::Device& device,
                         WGPUQueue queue, const float* hdr_rgba, uint32_t width, uint32_t height,
                         UpAxis up_axis = UpAxis::Y);

    /// Set uniform color environment (no texture, just dome color * intensity).
    /// Creates 1x1 cubemaps filled with the color. No pipelines needed.
    void set_uniform_environment(const webgpu::Device& device, WGPUQueue queue, float r, float g,
                                 float b);

    bool is_ready() const noexcept;

    WGPUTextureView prefiltered_env_view() const noexcept;
    WGPUTextureView env_cubemap_view() const noexcept;
    [[nodiscard]] WGPUTexture env_cubemap_texture() const noexcept {
        return m_env_cubemap;
    }
    WGPUTextureView irradiance_view() const noexcept;

   private:
    void release();

    void convert_equirect_to_cubemap(const IblPipelines& pipelines, const webgpu::Device& device,
                                     WGPUQueue queue, WGPUTexture equirect, UpAxis up_axis);
    void generate_env_mipmaps(const IblPipelines& pipelines, const webgpu::Device& device,
                              WGPUQueue queue);
    void convolve_irradiance(const IblPipelines& pipelines, const webgpu::Device& device,
                             WGPUQueue queue);
    void prefilter_specular(const IblPipelines& pipelines, const webgpu::Device& device,
                            WGPUQueue queue);

    WGPUTexture m_env_cubemap = nullptr;
    WGPUTexture m_prefiltered = nullptr;
    WGPUTexture m_irradiance = nullptr;
    WGPUTextureView m_env_cube_view = nullptr;
    WGPUTextureView m_prefiltered_view = nullptr;
    WGPUTextureView m_irradiance_view = nullptr;

    bool m_env_ready = false;
};

}  // namespace pts::rendering
