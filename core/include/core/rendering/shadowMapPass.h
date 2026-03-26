#pragma once

#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <variant>
#include <vector>

namespace pts::rendering {

inline constexpr uint32_t k_max_shadow_maps = 4;
inline constexpr uint32_t k_default_shadow_resolution = 2048;

/// Renders depth maps for shadow-casting distant lights.
class ShadowMapPass final : public IRenderPass {
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
    void add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) override;

    [[nodiscard]] WGPUTextureView shadow_array_view() const;

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

    // Shadow texture array (managed by this pass, not FrameGraph)
    WGPUTexture m_shadow_texture = nullptr;
    WGPUTextureView m_shadow_array_view = nullptr;      // full array view for sampling
    std::vector<WGPUTextureView> m_shadow_layer_views;  // per-layer views for rendering
    uint32_t m_current_layer_count = 0;

    uint32_t m_resolution = k_default_shadow_resolution;

    void ensure_shadow_texture(const webgpu::Device& device, uint32_t layer_count);
    void release_shadow_texture();
};

}  // namespace pts::rendering
