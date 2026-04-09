#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <variant>

namespace pts::rendering {

class ShaderLoader;

/// Screen-space contact shadow pass.
/// Reads scene_depth (Depth32Float), scene_normals (RG16Float), and the light
/// buffer, writes contact_shadow (R8Unorm, 1=lit, 0=shadowed) by ray-marching
/// the depth buffer toward each non-dome light.
class ContactShadowPass final : public IPass {
   public:
    explicit ContactShadowPass(const ShaderLoader& sl);
    ~ContactShadowPass() override;

    ContactShadowPass(const ContactShadowPass&) = delete;
    ContactShadowPass& operator=(const ContactShadowPass&) = delete;
    ContactShadowPass(ContactShadowPass&&) = delete;
    ContactShadowPass& operator=(ContactShadowPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "contact_shadow";
    }
    [[nodiscard]] auto is_ready() const noexcept -> bool override;
    [[nodiscard]] auto debug_targets() const noexcept
        -> std::pair<const DebugTarget*, uint32_t> override;

    struct Inputs {
        ResourceHandle depth;
        ResourceHandle normals;
        WGPUBuffer light_buffer;
        uint64_t light_buffer_size;
    };
    struct Outputs {
        ResourceHandle contact_shadow;
    };

    void do_setup(const webgpu::Device& device) override;
    Outputs add_to_frame_graph(FrameGraph& fg, const PassContext& ctx, const Inputs& in);
    void draw_imgui() override;

    // Tunable parameters (exposed via ImGui)
    bool m_enabled = true;
    float m_max_distance = 0.5f;
    float m_thickness = 0.05f;
    float m_normal_offset = 0.01f;
    int m_step_count = 16;

   private:
    struct Ready {
        webgpu::ShaderModule shader;
        webgpu::RenderPipeline pipeline;
        WGPUBindGroupLayout bgl = nullptr;

        // Samplers
        WGPUSampler depth_sampler = nullptr;   // non-filtering
        WGPUSampler linear_sampler = nullptr;  // linear filtering
    };

    void release_raw_handles();

    std::variant<std::monostate, Ready> m_state;
};

}  // namespace pts::rendering
