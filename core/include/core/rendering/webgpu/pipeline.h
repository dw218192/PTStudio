#pragma once

#include <core/rendering/webgpu/webgpu.h>

namespace pts::webgpu {

/// RAII wrapper for WGPUPipelineLayout.
/// Invariant: handle is non-null after construction (moved-from state is null but unusable).
class PipelineLayout {
   public:
    explicit PipelineLayout(WGPUPipelineLayout layout);

    PipelineLayout(const PipelineLayout&) = delete;
    auto operator=(const PipelineLayout&) -> PipelineLayout& = delete;

    PipelineLayout(PipelineLayout&& other) noexcept;
    auto operator=(PipelineLayout&& other) noexcept -> PipelineLayout&;

    ~PipelineLayout();

    [[nodiscard]] auto handle() const noexcept -> WGPUPipelineLayout;

   private:
    WGPUPipelineLayout m_layout;
};

/// RAII wrapper for WGPURenderPipeline.
/// Invariant: handle is non-null after construction (moved-from state is null but unusable).
class RenderPipeline {
   public:
    explicit RenderPipeline(WGPURenderPipeline pipeline);

    RenderPipeline(const RenderPipeline&) = delete;
    auto operator=(const RenderPipeline&) -> RenderPipeline& = delete;

    RenderPipeline(RenderPipeline&& other) noexcept;
    auto operator=(RenderPipeline&& other) noexcept -> RenderPipeline&;

    ~RenderPipeline();

    [[nodiscard]] auto handle() const noexcept -> WGPURenderPipeline;

   private:
    WGPURenderPipeline m_pipeline;
};

}  // namespace pts::webgpu
