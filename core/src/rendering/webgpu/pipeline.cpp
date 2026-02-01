#include <core/diagnostics.h>
#include <core/rendering/webgpu/pipeline.h>

namespace pts::webgpu {

// PipelineLayout implementation

PipelineLayout::PipelineLayout(WGPUPipelineLayout layout) : m_layout(layout) {
    INVARIANT_MSG(m_layout != nullptr, "handle is null");
}

PipelineLayout::PipelineLayout(PipelineLayout&& other) noexcept : m_layout(other.m_layout) {
    other.m_layout = nullptr;
}

auto PipelineLayout::operator=(PipelineLayout&& other) noexcept -> PipelineLayout& {
    if (this != &other) {
        if (m_layout != nullptr) {
            wgpuPipelineLayoutRelease(m_layout);
        }
        m_layout = other.m_layout;
        other.m_layout = nullptr;
    }
    return *this;
}

PipelineLayout::~PipelineLayout() {
    if (m_layout != nullptr) {
        wgpuPipelineLayoutRelease(m_layout);
    }
}

auto PipelineLayout::handle() const noexcept -> WGPUPipelineLayout {
    ASSERT_MSG(m_layout != nullptr, "use after move");
    return m_layout;
}

// RenderPipeline implementation

RenderPipeline::RenderPipeline(WGPURenderPipeline pipeline) : m_pipeline(pipeline) {
    INVARIANT_MSG(m_pipeline != nullptr, "handle is null");
}

RenderPipeline::RenderPipeline(RenderPipeline&& other) noexcept : m_pipeline(other.m_pipeline) {
    other.m_pipeline = nullptr;
}

auto RenderPipeline::operator=(RenderPipeline&& other) noexcept -> RenderPipeline& {
    if (this != &other) {
        if (m_pipeline != nullptr) {
            wgpuRenderPipelineRelease(m_pipeline);
        }
        m_pipeline = other.m_pipeline;
        other.m_pipeline = nullptr;
    }
    return *this;
}

RenderPipeline::~RenderPipeline() {
    if (m_pipeline != nullptr) {
        wgpuRenderPipelineRelease(m_pipeline);
    }
}

auto RenderPipeline::handle() const noexcept -> WGPURenderPipeline {
    ASSERT_MSG(m_pipeline != nullptr, "use after move");
    return m_pipeline;
}

}  // namespace pts::webgpu
