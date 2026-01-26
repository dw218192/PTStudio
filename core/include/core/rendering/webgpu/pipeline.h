#pragma once

#include <core/rendering/webgpu/webgpu.h>

namespace pts::webgpu {

class RenderPipeline {
   public:
    RenderPipeline() = default;
    explicit RenderPipeline(WGPURenderPipeline pipeline);

    RenderPipeline(const RenderPipeline&) = delete;
    auto operator=(const RenderPipeline&) -> RenderPipeline& = delete;

    RenderPipeline(RenderPipeline&& other) noexcept;
    auto operator=(RenderPipeline&& other) noexcept -> RenderPipeline&;

    ~RenderPipeline();

    [[nodiscard]] auto is_valid() const noexcept -> bool;
    [[nodiscard]] auto handle() const noexcept -> WGPURenderPipeline;

   private:
    WGPURenderPipeline m_pipeline = nullptr;
};

}  // namespace pts::webgpu
