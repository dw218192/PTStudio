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
        WGPUBindGroupLayout bind_group_layout = nullptr;
        rendering::LtcTextures ltc_textures;
        // Shadow receiver resources (bind group 1)
        WGPUBindGroupLayout shadow_recv_bgl = nullptr;
        WGPUSampler shadow_sampler = nullptr;
        // IBL resources (bind group 2)
        WGPUBindGroupLayout ibl_bgl = nullptr;
        WGPUSampler ibl_sampler = nullptr;
        // 1x1 black fallback textures for when IBL is not yet ready
        WGPUTexture fallback_cube_tex = nullptr;
        WGPUTextureView fallback_cube_view = nullptr;
        WGPUTexture fallback_2d_tex = nullptr;
        WGPUTextureView fallback_2d_view = nullptr;
        // Contact shadow resources (bind group 3)
        WGPUBindGroupLayout cs_bgl = nullptr;
        WGPUSampler cs_sampler = nullptr;
        WGPUTexture fallback_cs_tex = nullptr;
        WGPUTextureView fallback_cs_view = nullptr;
        // Skybox
        webgpu::ShaderModule skybox_shader;
        webgpu::RenderPipeline skybox_pipeline;
        WGPUBindGroupLayout skybox_bgl = nullptr;
    };

    std::variant<std::monostate, Ready> m_state;
};

}  // namespace pts::editor
