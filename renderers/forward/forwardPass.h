#pragma once

#include <core/rendering/ltcTextures.h>
#include <core/rendering/renderer.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <string_view>
#include <variant>

namespace pts::editor {

class ForwardPass final : public rendering::IRenderer {
   public:
    explicit ForwardPass(const rendering::ShaderLoader& sl);
    ~ForwardPass() override;

    ForwardPass(const ForwardPass&) = delete;
    ForwardPass& operator=(const ForwardPass&) = delete;
    ForwardPass(ForwardPass&&) = delete;
    ForwardPass& operator=(ForwardPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto is_ready() const noexcept -> bool override;
    [[nodiscard]] auto renderer_debug_targets() const noexcept
        -> std::pair<const DebugTarget*, uint32_t> override;

    void do_renderer_setup(const webgpu::Device& device) override;
    HdrOutputs do_add_to_frame_graph(rendering::FrameGraph& fg,
                                     const rendering::PassContext& ctx) override;
    static constexpr uint32_t k_uniform_align = 256;

   private:
    struct Ready {
        webgpu::ShaderModule shader;
        webgpu::RenderPipeline pipeline;
        WGPUBindGroupLayout descriptor_layout = nullptr;
        rendering::LtcTextures ltc_textures;
        // IBL resources (descriptor 2)
        WGPUBindGroupLayout ibl_desc_layout = nullptr;
        WGPUSampler ibl_sampler = nullptr;
        // 1x1 black fallback textures for when IBL is not yet ready
        WGPUTexture fallback_cube_tex = nullptr;
        WGPUTextureView fallback_cube_view = nullptr;
        WGPUTexture fallback_2d_tex = nullptr;
        WGPUTextureView fallback_2d_view = nullptr;
        // Skybox
        webgpu::ShaderModule skybox_shader;
        webgpu::RenderPipeline skybox_pipeline;
        WGPUBindGroupLayout skybox_desc_layout = nullptr;
    };

    std::variant<std::monostate, Ready> m_state;
};

}  // namespace pts::editor
