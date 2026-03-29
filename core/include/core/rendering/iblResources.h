#pragma once

#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <optional>

namespace pts::webgpu {
class Device;
}

namespace pts::rendering {

/// GPU resources for split-sum IBL: BRDF LUT, environment cubemap,
/// irradiance map, and specular prefilter map.
/// Call init() once per device lifetime, then set_environment() or
/// set_uniform_environment() to populate cubemaps.
class IblResources {
   public:
    IblResources() = default;
    ~IblResources();

    IblResources(const IblResources&) = delete;
    IblResources& operator=(const IblResources&) = delete;
    IblResources(IblResources&&) noexcept;
    IblResources& operator=(IblResources&&) noexcept;

    /// Create compute pipelines and generate the BRDF LUT.
    void init(const webgpu::Device& device, WGPUQueue queue);

    /// Set environment from equirectangular HDR pixel data (RGBA float32).
    /// Triggers equirect->cube, irradiance convolution, specular prefilter.
    void set_environment(const webgpu::Device& device, WGPUQueue queue, const float* hdr_rgba,
                         uint32_t width, uint32_t height);

    /// Set uniform color environment (no texture, just dome color * intensity).
    /// Creates 1x1 cubemaps filled with the color.
    void set_uniform_environment(const webgpu::Device& device, WGPUQueue queue, float r, float g,
                                 float b);

    /// True after init() + set_*environment().
    bool is_ready() const noexcept;

    WGPUTextureView prefiltered_env_view() const noexcept;
    WGPUTextureView irradiance_view() const noexcept;
    WGPUTextureView brdf_lut_view() const noexcept;
    WGPUSampler sampler() const noexcept;

    static constexpr uint32_t k_env_size = 256;
    static constexpr uint32_t k_irradiance_size = 32;
    static constexpr uint32_t k_brdf_lut_size = 512;
    static constexpr uint32_t k_prefilter_mip_count = 6;

   private:
    void release_env();
    void release_all();

    void generate_brdf_lut(const webgpu::Device& device, WGPUQueue queue);
    void convert_equirect_to_cubemap(const webgpu::Device& device, WGPUQueue queue,
                                     WGPUTexture equirect);
    void convolve_irradiance(const webgpu::Device& device, WGPUQueue queue);
    void prefilter_specular(const webgpu::Device& device, WGPUQueue queue);

    // Textures
    WGPUTexture m_env_cubemap = nullptr;
    WGPUTexture m_irradiance = nullptr;
    WGPUTexture m_brdf_lut = nullptr;
    WGPUTextureView m_env_cube_view = nullptr;
    WGPUTextureView m_irradiance_view = nullptr;
    WGPUTextureView m_brdf_lut_view = nullptr;
    WGPUSampler m_sampler = nullptr;

    // Compute pipelines
    std::optional<webgpu::ComputePipeline> m_equirect_to_cube_pipeline;
    std::optional<webgpu::ComputePipeline> m_irradiance_pipeline;
    std::optional<webgpu::ComputePipeline> m_prefilter_pipeline;
    std::optional<webgpu::ComputePipeline> m_brdf_lut_pipeline;

    // Bind group layouts
    WGPUBindGroupLayout m_equirect_bgl = nullptr;
    WGPUBindGroupLayout m_convolve_bgl = nullptr;
    WGPUBindGroupLayout m_brdf_lut_bgl = nullptr;

    bool m_initialized = false;
    bool m_env_ready = false;
};

}  // namespace pts::rendering
