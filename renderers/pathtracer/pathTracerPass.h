#pragma once

#include <core/rendering/renderer.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <string_view>
#include <variant>

namespace pts::editor {

class PathTracerPass final : public rendering::IRenderer {
   public:
    using IRenderer::IRenderer;
    ~PathTracerPass() override;

    PathTracerPass(const PathTracerPass&) = delete;
    PathTracerPass& operator=(const PathTracerPass&) = delete;
    PathTracerPass(PathTracerPass&&) = delete;
    PathTracerPass& operator=(PathTracerPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto is_ready() const noexcept -> bool override;

    void do_renderer_setup(const webgpu::Device& device) override;
    void do_draw_imgui() override;
    void draw_viewport_overlay(const ViewportOverlayParams& params) override;
    HdrOutputs do_add_to_frame_graph(rendering::FrameGraph& fg,
                                     const rendering::PassContext& ctx) override;
    void draw_viewport_controls() override;

   private:
    void ensure_pixel_buffers(const webgpu::Device& device, uint32_t width, uint32_t height);

    struct Ready {
        webgpu::ShaderModule compute_shader;
        webgpu::ComputePipeline compute_pipeline;
        webgpu::Buffer uniform_buffer;
        WGPUBindGroupLayout compute_desc_layout = nullptr;
        WGPUBindGroupLayout ibl_desc_layout = nullptr;

        webgpu::ShaderModule blit_shader;
        webgpu::RenderPipeline blit_pipeline;
        WGPUBindGroupLayout blit_desc_layout = nullptr;
    };

    std::variant<std::monostate, Ready> m_state;

    // Per-pixel buffers
    webgpu::Buffer m_accum_buffer;
    webgpu::Buffer m_output_buffer;
    uint32_t m_pixel_width = 0;
    uint32_t m_pixel_height = 0;

    // Camera / scene change detection
    glm::mat4 m_prev_vp{0.0f};
    WGPUBuffer m_prev_instance_handle = nullptr;
    uint32_t m_prev_light_version = UINT32_MAX;
    uint32_t m_frame_count = 0;
};

}  // namespace pts::editor
