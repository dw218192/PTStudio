#pragma once

#include <core/rendering/renderer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <string_view>
#include <variant>

namespace pts::editor {

class WireframePass final : public rendering::IRenderer {
   public:
    using IRenderer::IRenderer;
    ~WireframePass() override;

    WireframePass(const WireframePass&) = delete;
    WireframePass& operator=(const WireframePass&) = delete;
    WireframePass(WireframePass&&) = delete;
    WireframePass& operator=(WireframePass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto is_ready() const noexcept -> bool override;

    void do_renderer_setup(const webgpu::Device& device) override;
    HdrOutputs do_add_to_frame_graph(rendering::FrameGraph& fg,
                                     const rendering::PassContext& ctx) override;

    static constexpr uint32_t k_uniform_align = 256;

   private:
    struct Ready {
        webgpu::ShaderModule shader;
        webgpu::RenderPipeline pipeline;
        WGPUBindGroupLayout descriptor_layout = nullptr;
    };

    std::variant<std::monostate, Ready> m_state;
};

}  // namespace pts::editor
