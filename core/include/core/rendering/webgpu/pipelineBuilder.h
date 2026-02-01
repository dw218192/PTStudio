#pragma once

#include <core/rendering/webgpu/device.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <string>
#include <string_view>

namespace pts::webgpu {

class RenderPipelineBuilder {
   public:
    explicit RenderPipelineBuilder(const Device& device);

    auto shader(const ShaderModule& module) -> RenderPipelineBuilder&;
    auto vertex_entry(std::string_view name) -> RenderPipelineBuilder&;
    auto fragment_entry(std::string_view name) -> RenderPipelineBuilder&;
    auto color_format(WGPUTextureFormat format) -> RenderPipelineBuilder&;
    auto topology(WGPUPrimitiveTopology topo) -> RenderPipelineBuilder&;
    auto cull_mode(WGPUCullMode mode) -> RenderPipelineBuilder&;
    auto front_face(WGPUFrontFace face) -> RenderPipelineBuilder&;
    auto blend_state(const WGPUBlendState& blend) -> RenderPipelineBuilder&;
    auto depth_format(WGPUTextureFormat format) -> RenderPipelineBuilder&;
    auto depth_write(bool enabled) -> RenderPipelineBuilder&;
    auto depth_compare(WGPUCompareFunction func) -> RenderPipelineBuilder&;
    auto sample_count(uint32_t count) -> RenderPipelineBuilder&;

    [[nodiscard]] auto build() const -> RenderPipeline;

   private:
    const Device& m_device;
    WGPUShaderModule m_shader_module = nullptr;
    std::string m_vertex_entry = "vs_main";
    std::string m_fragment_entry = "fs_main";
    WGPUTextureFormat m_color_format = WGPUTextureFormat_BGRA8Unorm;
    WGPUPrimitiveTopology m_topology = WGPUPrimitiveTopology_TriangleList;
    WGPUCullMode m_cull_mode = WGPUCullMode_None;
    WGPUFrontFace m_front_face = WGPUFrontFace_CCW;
    WGPUBlendState m_blend_state = {};
    bool m_has_blend = false;
    WGPUTextureFormat m_depth_format = WGPUTextureFormat_Undefined;
    bool m_depth_write = false;
    WGPUCompareFunction m_depth_compare = WGPUCompareFunction_Always;
    uint32_t m_sample_count = 1;
};

}  // namespace pts::webgpu
