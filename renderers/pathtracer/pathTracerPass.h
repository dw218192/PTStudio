#pragma once

#include <core/rendering/renderer.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <string_view>

namespace pts::editor {

class PathTracerPass final : public rendering::IRenderer {
   public:
    using IRenderer::IRenderer;

    PathTracerPass(const PathTracerPass&) = delete;
    PathTracerPass& operator=(const PathTracerPass&) = delete;
    PathTracerPass(PathTracerPass&&) = delete;
    PathTracerPass& operator=(PathTracerPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;

    void do_draw_imgui() override;
    void draw_viewport_overlay(const ViewportOverlayParams& params) override;
    HdrOutputs do_add_to_frame_graph(rendering::FrameGraph& fg,
                                     const rendering::PassContext& ctx) override;
    void draw_viewport_controls() override;

   private:
    void ensure_pixel_buffers(const webgpu::Device& device, uint32_t width, uint32_t height);

    // Per-pixel buffers
    webgpu::Buffer m_uniform_buffer;
    webgpu::Buffer m_accum_buffer;
    webgpu::Buffer m_output_buffer;
    uint64_t m_output_buffer_version = 0;  // bumped when m_output_buffer is recreated
    uint32_t m_pixel_width = 0;
    uint32_t m_pixel_height = 0;

    // Camera / scene change detection
    glm::mat4 m_prev_vp{0.0f};
    WGPUBuffer m_prev_instance_handle = nullptr;
    uint64_t m_prev_light_version = UINT64_MAX;
    uint32_t m_frame_count = 0;
};

}  // namespace pts::editor
