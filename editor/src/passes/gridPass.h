#pragma once

#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <string_view>
#include <variant>

namespace pts::editor {

class GridPass final : public rendering::IPass {
   public:
    using IPass::IPass;
    ~GridPass() override;

    GridPass(const GridPass&) = delete;
    GridPass& operator=(const GridPass&) = delete;
    GridPass(GridPass&&) = delete;
    GridPass& operator=(GridPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto is_ready() const noexcept -> bool override;

    void do_setup(const webgpu::Device& device) override;
    void render(rendering::FrameGraph& fg, const rendering::PassContext& ctx,
                rendering::TextureHandle color, rendering::TextureHandle depth);

   private:
    struct Ready {
        webgpu::ShaderModule shader;
        webgpu::RenderPipeline pipeline;
        WGPUBindGroupLayout descriptor_layout = nullptr;
    };

    std::variant<std::monostate, Ready> m_state;
};

}  // namespace pts::editor
