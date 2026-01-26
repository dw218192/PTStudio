#include <core/rendering/webgpu/pipeline.h>

namespace pts::webgpu {

RenderPipeline::RenderPipeline(WGPURenderPipeline pipeline) : m_pipeline(pipeline) {
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

auto RenderPipeline::is_valid() const noexcept -> bool {
    return m_pipeline != nullptr;
}

auto RenderPipeline::handle() const noexcept -> WGPURenderPipeline {
    return m_pipeline;
}

}  // namespace pts::webgpu
