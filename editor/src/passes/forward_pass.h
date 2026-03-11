#pragma once

#include <core/rendering/scenePass.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <optional>
#include <string_view>
#include <variant>

namespace pts::editor {

class ForwardPass final : public rendering::IScenePass {
   public:
    ForwardPass() = default;
    ~ForwardPass() override;

    ForwardPass(const ForwardPass&) = delete;
    ForwardPass& operator=(const ForwardPass&) = delete;
    ForwardPass(ForwardPass&&) = delete;
    ForwardPass& operator=(ForwardPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto is_ready() const noexcept -> bool override;

    void setup(const webgpu::Device& device) override;
    void add_to_frame_graph(rendering::FrameGraph& fg,
                            const rendering::PassContext& ctx) override;

   private:
    struct Ready {
        webgpu::ShaderModule shader;
        webgpu::RenderPipeline pipeline;
        webgpu::Buffer uniform_buffer;
        WGPUBindGroup bind_group = nullptr;
        WGPUBindGroupLayout bind_group_layout = nullptr;
    };

    std::variant<std::monostate, Ready> m_state;
};

}  // namespace pts::editor
