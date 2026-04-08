#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/toneMappingPass.h>

#include <memory>
#include <optional>
#include <vector>

namespace pts::rendering {

class IRenderer : public IPass {
   public:
    using IPass::IPass;
    ~IRenderer() override;

    struct Outputs {
        TextureHandle color;                 // tone-mapped LDR, display-ready
        TextureHandle hdr_color;             // raw HDR scene color (for editor overlays)
        std::optional<TextureHandle> depth;  // compute-only renderers may not produce
    };

    /// Public entry point (non-virtual, NVI).
    /// Calls do_add_to_frame_graph → gets HDR scene color + depth,
    /// then runs tone mapping → LDR display-ready color.
    Outputs add_to_frame_graph(FrameGraph& fg, const PassContext& ctx);

    // Exposure controls (delegated to ToneMappingPass)
    float& exposure();
    uint32_t& tone_map_mode();
    bool& auto_exposure();
    float& adaptation_speed();

    template <typename T, typename... Args>
    T& add_pass(Args&&... args) {
        auto p = std::make_unique<T>(std::forward<Args>(args)...);
        auto& ref = *p;
        m_children.push_back(std::move(p));
        return ref;
    }

    template <typename T>
    T* get_pass() const {
        for (auto& c : m_children) {
            if (auto* p = dynamic_cast<T*>(c.get())) return p;
        }
        return nullptr;
    }

    // ── Lifecycle: auto-forwarded to all children ──

    void on_shaders_reloaded(const webgpu::Device& device) override;
    void draw_imgui() override;

    void draw_viewport_overlay(const ViewportOverlayParams& params) override {
        for (auto& c : m_children) c->draw_viewport_overlay(params);
    }

    void update_texture_refs(FrameGraph& fg) override {
        for (auto& c : m_children) c->update_texture_refs(fg);
    }

    /// Aggregated debug targets: this renderer's own + all children's.
    [[nodiscard]] auto debug_targets() const noexcept
        -> std::pair<const DebugTarget*, uint32_t> override {
        return {m_all_debug_targets.data(), static_cast<uint32_t>(m_all_debug_targets.size())};
    }

   protected:
    /// What do_add_to_frame_graph returns — HDR color before tone mapping.
    struct HdrOutputs {
        TextureHandle color;                 // HDR scene color
        std::optional<TextureHandle> depth;  // compute-only renderers may not produce
        std::optional<TextureHandle> ssao;   // ambient occlusion (if available)
    };

    void do_setup(const webgpu::Device& device) override;

    virtual void do_renderer_setup(const webgpu::Device& device) = 0;
    virtual HdrOutputs do_add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) = 0;
    virtual void do_draw_imgui() {};

    /// Renderer's own debug targets (not children's). Override instead of debug_targets().
    [[nodiscard]] virtual auto renderer_debug_targets() const noexcept
        -> std::pair<const DebugTarget*, uint32_t> {
        return {nullptr, 0};
    }

   private:
    void collect_debug_targets();

    std::unique_ptr<ToneMappingPass> m_tonemapping;
    bool m_tonemapping_enabled = true;
    std::vector<std::unique_ptr<IPass>> m_children;
    std::vector<DebugTarget> m_all_debug_targets;
};

}  // namespace pts::rendering
