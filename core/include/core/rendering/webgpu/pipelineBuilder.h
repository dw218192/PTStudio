#pragma once

#include <core/rendering/webgpu/device.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pts::webgpu {

struct VertexBufferLayout {
    uint64_t stride;
    WGPUVertexStepMode step_mode = WGPUVertexStepMode_Vertex;
    std::vector<WGPUVertexAttribute> attributes;
};

class RenderPipelineBuilder {
   public:
    explicit RenderPipelineBuilder(const Device& device);

    auto shader(const ShaderModule& module) -> RenderPipelineBuilder&;
    auto vertex_entry(std::string_view name) -> RenderPipelineBuilder&;
    auto fragment_entry(std::string_view name) -> RenderPipelineBuilder&;
    auto color_format(WGPUTextureFormat format, uint32_t index = 0) -> RenderPipelineBuilder&;
    auto topology(WGPUPrimitiveTopology topo) -> RenderPipelineBuilder&;
    auto cull_mode(WGPUCullMode mode) -> RenderPipelineBuilder&;
    auto front_face(WGPUFrontFace face) -> RenderPipelineBuilder&;
    auto blend_state(const WGPUBlendState& blend, uint32_t index = 0) -> RenderPipelineBuilder&;
    auto depth_format(WGPUTextureFormat format) -> RenderPipelineBuilder&;
    auto depth_write(bool enabled) -> RenderPipelineBuilder&;
    auto depth_compare(WGPUCompareFunction func) -> RenderPipelineBuilder&;
    auto sample_count(uint32_t count) -> RenderPipelineBuilder&;
    auto vertex_buffer(VertexBufferLayout layout) -> RenderPipelineBuilder&;
    auto pipeline_layout(WGPUPipelineLayout layout) -> RenderPipelineBuilder&;
    auto no_fragment() -> RenderPipelineBuilder&;

    /// Configure vertex buffer layout from shader reflection metadata.
    /// T must have static constexpr stride, step_mode, and attributes members.
    template <typename VertexLayoutT>
    auto vertex_layout() -> RenderPipelineBuilder& {
        VertexBufferLayout layout;
        layout.stride = VertexLayoutT::stride;
        layout.step_mode = VertexLayoutT::step_mode;
        layout.attributes.reserve(VertexLayoutT::attributes.size());
        for (const auto& attr : VertexLayoutT::attributes) {
            layout.attributes.push_back(attr);
        }
        return vertex_buffer(std::move(layout));
    }

    [[nodiscard]] auto build() const -> RenderPipeline;

   private:
    /// Ensure m_color_targets and m_blend_states have at least (index + 1) entries,
    /// initializing new entries with defaults.
    void ensure_target_count(uint32_t index);

    const Device& m_device;
    WGPUShaderModule m_shader_module = nullptr;
    std::string m_vertex_entry = "vs_main";
    std::string m_fragment_entry = "fs_main";
    std::vector<WGPUColorTargetState> m_color_targets;
    std::vector<WGPUBlendState> m_blend_states;
    std::vector<bool> m_has_blend;
    WGPUPrimitiveTopology m_topology = WGPUPrimitiveTopology_TriangleList;
    WGPUCullMode m_cull_mode = WGPUCullMode_None;
    WGPUFrontFace m_front_face = WGPUFrontFace_CCW;
    WGPUTextureFormat m_depth_format = WGPUTextureFormat_Undefined;
    bool m_depth_write = false;
    WGPUCompareFunction m_depth_compare = WGPUCompareFunction_Always;
    uint32_t m_sample_count = 1;
    std::vector<VertexBufferLayout> m_vertex_buffers;
    WGPUPipelineLayout m_pipeline_layout = nullptr;
    bool m_has_fragment = true;
};

class ComputePipelineBuilder {
   public:
    explicit ComputePipelineBuilder(const Device& device);

    auto shader(const ShaderModule& module) -> ComputePipelineBuilder&;
    auto entry_point(std::string_view name) -> ComputePipelineBuilder&;
    auto pipeline_layout(WGPUPipelineLayout layout) -> ComputePipelineBuilder&;

    [[nodiscard]] auto build() const -> ComputePipeline;

   private:
    const Device& m_device;
    WGPUShaderModule m_shader = nullptr;
    std::string m_entry_point = "cs_main";
    WGPUPipelineLayout m_layout = nullptr;
};

}  // namespace pts::webgpu
