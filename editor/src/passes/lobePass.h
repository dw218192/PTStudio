#pragma once

#include <core/rendering/scenePass.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <string_view>
#include <variant>

namespace pts::editor {

class LobePass final : public rendering::IScenePass {
   public:
    LobePass() = default;
    ~LobePass() override;

    LobePass(const LobePass&) = delete;
    LobePass& operator=(const LobePass&) = delete;
    LobePass(LobePass&&) = delete;
    LobePass& operator=(LobePass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto is_ready() const noexcept -> bool override;

    void setup(const webgpu::Device& device) override;
    void add_to_frame_graph(rendering::FrameGraph& fg, const rendering::PassContext& ctx) override;

    /// Draw the ImGui controls panel.
    void draw_imgui_controls();

    static constexpr uint32_t k_texture_size = 256;
    static constexpr uint32_t k_grid_cols = 128;
    static constexpr uint32_t k_grid_rows = 64;

   private:
    struct Ready {
        webgpu::ShaderModule shader;
        webgpu::RenderPipeline pipeline;
        webgpu::Buffer uniform_buffer;
        WGPUBindGroup bind_group = nullptr;
        WGPUBindGroupLayout bind_group_layout = nullptr;
    };

    std::variant<std::monostate, Ready> m_state;

    // ImGui parameters
    float m_roughness = 0.5f;
    float m_metallic = 0.0f;
    float m_scale = 1.0f;
    float m_light_azimuth_deg = 0.0f;
    float m_light_elevation_deg = 45.0f;
};

}  // namespace pts::editor
