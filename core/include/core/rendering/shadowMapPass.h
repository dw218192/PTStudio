#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/outputLayout.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <variant>

namespace pts::rendering {

inline constexpr uint32_t k_max_shadow_maps = 4;
inline constexpr uint32_t k_default_shadow_resolution = 2048;

/// Renders depth maps for shadow-casting distant lights.
class ShadowMapPass final : public IPass {
   public:
    explicit ShadowMapPass(const ShaderLoader& sl);
    ~ShadowMapPass() override;

    ShadowMapPass(const ShadowMapPass&) = delete;
    ShadowMapPass& operator=(const ShadowMapPass&) = delete;
    ShadowMapPass(ShadowMapPass&&) = delete;
    ShadowMapPass& operator=(ShadowMapPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "shadow_map";
    }
    [[nodiscard]] auto is_ready() const noexcept -> bool override;
    [[nodiscard]] auto requires_viewport() const noexcept -> bool override {
        return false;
    }

    void do_setup(const webgpu::Device& device) override;
    void draw_imgui() override;

    struct Inputs {};
    struct Outputs {
        TextureHandle shadow_array;
        BufferHandle shadow_info;
        DescriptorHandle consumer_desc;
    };
    Outputs add_to_frame_graph(FrameGraph& fg, const PassContext& ctx, const Inputs&);

    /// Layout for the consumer bind group (shadow receiver). Non-owning.
    [[nodiscard]] WGPUBindGroupLayout consumer_layout() const;

    [[nodiscard]] bool enabled() const {
        return m_enabled;
    }

   private:
    bool m_enabled = true;
    static constexpr uint32_t k_uniform_align = 256;

    struct Ready {
        webgpu::ShaderModule shader;
        webgpu::RenderPipeline pipeline;
        WGPUBindGroupLayout desc_layout = nullptr;
        OutputLayoutInfo output_layout;
    };
    std::variant<std::monostate, Ready> m_state;

    uint32_t m_resolution = k_default_shadow_resolution;
};

}  // namespace pts::rendering
