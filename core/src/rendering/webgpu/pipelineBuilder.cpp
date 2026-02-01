#include <core/diagnostics.h>
#include <core/rendering/webgpu/pipelineBuilder.h>

#include <stdexcept>

namespace pts::webgpu {

RenderPipelineBuilder::RenderPipelineBuilder(const Device& device) : m_device(device) {
}

auto RenderPipelineBuilder::shader(const ShaderModule& module) -> RenderPipelineBuilder& {
    m_shader_module = module.handle();
    return *this;
}

auto RenderPipelineBuilder::vertex_entry(std::string_view name) -> RenderPipelineBuilder& {
    m_vertex_entry = std::string(name);
    return *this;
}

auto RenderPipelineBuilder::fragment_entry(std::string_view name) -> RenderPipelineBuilder& {
    m_fragment_entry = std::string(name);
    return *this;
}

auto RenderPipelineBuilder::color_format(WGPUTextureFormat format) -> RenderPipelineBuilder& {
    m_color_format = format;
    return *this;
}

auto RenderPipelineBuilder::topology(WGPUPrimitiveTopology topo) -> RenderPipelineBuilder& {
    m_topology = topo;
    return *this;
}

auto RenderPipelineBuilder::cull_mode(WGPUCullMode mode) -> RenderPipelineBuilder& {
    m_cull_mode = mode;
    return *this;
}

auto RenderPipelineBuilder::front_face(WGPUFrontFace face) -> RenderPipelineBuilder& {
    m_front_face = face;
    return *this;
}

auto RenderPipelineBuilder::blend_state(const WGPUBlendState& blend) -> RenderPipelineBuilder& {
    m_blend_state = blend;
    m_has_blend = true;
    return *this;
}

auto RenderPipelineBuilder::depth_format(WGPUTextureFormat format) -> RenderPipelineBuilder& {
    m_depth_format = format;
    return *this;
}

auto RenderPipelineBuilder::depth_write(bool enabled) -> RenderPipelineBuilder& {
    m_depth_write = enabled;
    return *this;
}

auto RenderPipelineBuilder::depth_compare(WGPUCompareFunction func) -> RenderPipelineBuilder& {
    m_depth_compare = func;
    return *this;
}

auto RenderPipelineBuilder::sample_count(uint32_t count) -> RenderPipelineBuilder& {
    m_sample_count = count;
    return *this;
}

auto RenderPipelineBuilder::build() const -> RenderPipeline {
    PRECONDITION_MSG(m_shader_module != nullptr, "shader module not set");

    // Create empty pipeline layout
    PipelineLayout layout = m_device.create_pipeline_layout();

    // Vertex state
    WGPUVertexState vertex_state = {};
    vertex_state.module = m_shader_module;
    vertex_state.entryPoint = WGPUStringView{m_vertex_entry.c_str(), m_vertex_entry.size()};
    vertex_state.bufferCount = 0;
    vertex_state.buffers = nullptr;

    // Color target state
    WGPUColorTargetState color_target = {};
    color_target.format = m_color_format;
    color_target.writeMask = WGPUColorWriteMask_All;
    color_target.blend = m_has_blend ? &m_blend_state : nullptr;

    // Fragment state
    WGPUFragmentState fragment_state = {};
    fragment_state.module = m_shader_module;
    fragment_state.entryPoint = WGPUStringView{m_fragment_entry.c_str(), m_fragment_entry.size()};
    fragment_state.targetCount = 1;
    fragment_state.targets = &color_target;

    // Primitive state with sensible defaults
    WGPUPrimitiveState primitive_state = {};
    primitive_state.topology = m_topology;
    primitive_state.stripIndexFormat = WGPUIndexFormat_Undefined;
    primitive_state.frontFace = m_front_face;
    primitive_state.cullMode = m_cull_mode;

    // Multisample state with sensible defaults
    WGPUMultisampleState multisample_state = {};
    multisample_state.count = m_sample_count;
    multisample_state.mask = 0xFFFFFFFFu;
    multisample_state.alphaToCoverageEnabled = false;

    // Depth stencil state (optional)
    WGPUDepthStencilState depth_stencil_state = {};
    depth_stencil_state.format = m_depth_format;
    depth_stencil_state.depthWriteEnabled =
        m_depth_write ? WGPUOptionalBool_True : WGPUOptionalBool_False;
    depth_stencil_state.depthCompare = m_depth_compare;
    depth_stencil_state.stencilFront.compare = WGPUCompareFunction_Always;
    depth_stencil_state.stencilFront.failOp = WGPUStencilOperation_Keep;
    depth_stencil_state.stencilFront.depthFailOp = WGPUStencilOperation_Keep;
    depth_stencil_state.stencilFront.passOp = WGPUStencilOperation_Keep;
    depth_stencil_state.stencilBack = depth_stencil_state.stencilFront;

    // Pipeline descriptor
    WGPURenderPipelineDescriptor pipeline_desc = {};
    pipeline_desc.layout = layout.handle();
    pipeline_desc.vertex = vertex_state;
    pipeline_desc.fragment = &fragment_state;
    pipeline_desc.primitive = primitive_state;
    pipeline_desc.multisample = multisample_state;
    pipeline_desc.depthStencil =
        (m_depth_format != WGPUTextureFormat_Undefined) ? &depth_stencil_state : nullptr;

    WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(m_device.handle(), &pipeline_desc);

    if (pipeline == nullptr) {
        throw std::runtime_error("RenderPipelineBuilder: failed to create render pipeline");
    }

    return RenderPipeline(pipeline);
}

}  // namespace pts::webgpu
