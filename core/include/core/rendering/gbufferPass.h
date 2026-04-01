#pragma once

#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <variant>

namespace pts::rendering {

class ShaderLoader;

/// Renders view-space normals and depth as a geometry pre-pass.
/// Added as a child pass of any renderer via add_pass<GBufferPass>(sl).
class GBufferPass final : public IRenderPass {
   public:
    explicit GBufferPass(const ShaderLoader& sl);
    ~GBufferPass() override;

    GBufferPass(const GBufferPass&) = delete;
    GBufferPass& operator=(const GBufferPass&) = delete;
    GBufferPass(GBufferPass&&) = delete;
    GBufferPass& operator=(GBufferPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "gbuffer";
    }
    [[nodiscard]] auto is_ready() const noexcept -> bool override;

    void do_setup(const webgpu::Device& device) override;
    void add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) override;

   private:
    static constexpr uint32_t k_uniform_align = 256;

    struct Ready {
        webgpu::ShaderModule shader;
        webgpu::RenderPipeline pipeline;
        webgpu::Buffer per_object_uniform_buf;
        WGPUBindGroupLayout bgl = nullptr;
        WGPUBindGroup bind_group = nullptr;
        uint32_t object_capacity = 0;
    };
    std::variant<std::monostate, Ready> m_state;
};

}  // namespace pts::rendering
