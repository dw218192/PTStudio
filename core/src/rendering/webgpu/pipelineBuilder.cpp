#include <core/diagnostics.h>
#include <core/rendering/webgpu/pipelineBuilder.h>

#include <optional>
#include <stdexcept>

namespace pts::webgpu {

// ---------------------------------------------------------------------------
// RenderPipelineBuilder
// ---------------------------------------------------------------------------

RenderPipelineBuilder::RenderPipelineBuilder(const Device& device) : m_device(device) {
    // Initialize with one default color target (BGRA8Unorm, write-all, no blend).
    WGPUColorTargetState default_target = {};
    default_target.format = WGPUTextureFormat_BGRA8Unorm;
    default_target.writeMask = WGPUColorWriteMask_All;
    default_target.blend = nullptr;

    m_color_targets.push_back(default_target);
    m_blend_states.push_back({});
    m_has_blend.push_back(false);
}

auto RenderPipelineBuilder::shader(const ShaderModule& module) -> RenderPipelineBuilder& {
    m_shader_module = module.handle();
    return *this;
}

auto RenderPipelineBuilder::shader(WGPUShaderModule module) -> RenderPipelineBuilder& {
    m_shader_module = module;
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

auto RenderPipelineBuilder::color_format(WGPUTextureFormat format, uint32_t index)
    -> RenderPipelineBuilder& {
    ensure_target_count(index);
    m_color_targets[index].format = format;
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

auto RenderPipelineBuilder::blend_state(const WGPUBlendState& blend, uint32_t index)
    -> RenderPipelineBuilder& {
    ensure_target_count(index);
    m_blend_states[index] = blend;
    m_has_blend[index] = true;
    return *this;
}

auto RenderPipelineBuilder::write_mask(WGPUColorWriteMask mask, uint32_t index)
    -> RenderPipelineBuilder& {
    ensure_target_count(index);
    m_color_targets[index].writeMask = mask;
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

auto RenderPipelineBuilder::depth_bias(int32_t constant, float slope_scale)
    -> RenderPipelineBuilder& {
    m_depth_bias = constant;
    m_depth_bias_slope_scale = slope_scale;
    return *this;
}

auto RenderPipelineBuilder::sample_count(uint32_t count) -> RenderPipelineBuilder& {
    m_sample_count = count;
    return *this;
}

auto RenderPipelineBuilder::vertex_buffer(VertexBufferLayout layout) -> RenderPipelineBuilder& {
    m_vertex_buffers.push_back(std::move(layout));
    return *this;
}

auto RenderPipelineBuilder::pipeline_layout(WGPUPipelineLayout layout) -> RenderPipelineBuilder& {
    m_pipeline_layout = layout;
    return *this;
}

auto RenderPipelineBuilder::no_fragment() -> RenderPipelineBuilder& {
    m_has_fragment = false;
    m_color_targets.clear();
    m_blend_states.clear();
    m_has_blend.clear();
    return *this;
}

void RenderPipelineBuilder::ensure_target_count(uint32_t index) {
    auto required = static_cast<size_t>(index) + 1;
    while (m_color_targets.size() < required) {
        WGPUColorTargetState target = {};
        target.format = WGPUTextureFormat_BGRA8Unorm;
        target.writeMask = WGPUColorWriteMask_All;
        target.blend = nullptr;

        m_color_targets.push_back(target);
        m_blend_states.push_back({});
        m_has_blend.push_back(false);
    }
}

auto RenderPipelineBuilder::build() const -> RenderPipeline {
    PRECONDITION_MSG(m_shader_module != nullptr, "shader module not set");

    // Use custom pipeline layout or create empty one
    std::optional<PipelineLayout> owned_layout;
    if (!m_pipeline_layout) {
        owned_layout = m_device.create_pipeline_layout();
    }
    WGPUPipelineLayout layout_handle =
        m_pipeline_layout ? m_pipeline_layout : owned_layout->handle();

    // Convert stored vertex buffer layouts to WGPUVertexBufferLayout array
    std::vector<WGPUVertexBufferLayout> wgpu_vertex_buffers(m_vertex_buffers.size());
    for (size_t i = 0; i < m_vertex_buffers.size(); i++) {
        wgpu_vertex_buffers[i] = {};
        wgpu_vertex_buffers[i].arrayStride = m_vertex_buffers[i].stride;
        wgpu_vertex_buffers[i].stepMode = m_vertex_buffers[i].step_mode;
        wgpu_vertex_buffers[i].attributeCount = m_vertex_buffers[i].attributes.size();
        wgpu_vertex_buffers[i].attributes = m_vertex_buffers[i].attributes.data();
    }

    // Vertex state
    WGPUVertexState vertex_state = {};
    vertex_state.module = m_shader_module;
    vertex_state.entryPoint.data = m_vertex_entry.c_str();
    vertex_state.entryPoint.length = m_vertex_entry.size();
    vertex_state.bufferCount = wgpu_vertex_buffers.size();
    vertex_state.buffers = wgpu_vertex_buffers.empty() ? nullptr : wgpu_vertex_buffers.data();

    // Build color targets with fixup of interior blend pointers.
    // Copy vectors so we can safely set pointer into local blend_states copy.
    auto color_targets = m_color_targets;
    auto blend_states = m_blend_states;
    WGPUFragmentState fragment_state = {};

    if (m_has_fragment) {
        for (size_t i = 0; i < color_targets.size(); i++) {
            color_targets[i].blend = m_has_blend[i] ? &blend_states[i] : nullptr;
        }

        // Fragment state
        fragment_state.module = m_shader_module;
        fragment_state.entryPoint.data = m_fragment_entry.c_str();
        fragment_state.entryPoint.length = m_fragment_entry.size();
        fragment_state.targetCount = static_cast<uint32_t>(color_targets.size());
        fragment_state.targets = color_targets.data();
    }

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
    depth_stencil_state.depthBias = m_depth_bias;
    depth_stencil_state.depthBiasSlopeScale = m_depth_bias_slope_scale;

    // Pipeline descriptor
    WGPURenderPipelineDescriptor pipeline_desc = {};
    pipeline_desc.layout = layout_handle;
    pipeline_desc.vertex = vertex_state;
    pipeline_desc.fragment = m_has_fragment ? &fragment_state : nullptr;
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

// ---------------------------------------------------------------------------
// ComputePipelineBuilder
// ---------------------------------------------------------------------------

ComputePipelineBuilder::ComputePipelineBuilder(const Device& device) : m_device(device) {
}

auto ComputePipelineBuilder::shader(const ShaderModule& module) -> ComputePipelineBuilder& {
    m_shader = module.handle();
    return *this;
}

auto ComputePipelineBuilder::shader(WGPUShaderModule module) -> ComputePipelineBuilder& {
    m_shader = module;
    return *this;
}

auto ComputePipelineBuilder::entry_point(std::string_view name) -> ComputePipelineBuilder& {
    m_entry_point = std::string(name);
    return *this;
}

auto ComputePipelineBuilder::pipeline_layout(WGPUPipelineLayout layout) -> ComputePipelineBuilder& {
    m_layout = layout;
    return *this;
}

auto ComputePipelineBuilder::build() const -> ComputePipeline {
    PRECONDITION_MSG(m_shader != nullptr, "shader module not set");

    // Use custom pipeline layout or create empty one
    std::optional<PipelineLayout> owned_layout;
    if (!m_layout) {
        owned_layout = m_device.create_pipeline_layout();
    }
    WGPUPipelineLayout layout_handle = m_layout ? m_layout : owned_layout->handle();

    WGPUComputePipelineDescriptor desc = {};
    desc.layout = layout_handle;
    desc.compute.module = m_shader;
    desc.compute.entryPoint.data = m_entry_point.c_str();
    desc.compute.entryPoint.length = m_entry_point.size();

    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(m_device.handle(), &desc);

    if (pipeline == nullptr) {
        throw std::runtime_error("ComputePipelineBuilder: failed to create compute pipeline");
    }

    return ComputePipeline(pipeline);
}

}  // namespace pts::webgpu
