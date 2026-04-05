#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>

#include <memory>
#include <vector>

namespace pts::rendering {

class IRenderer : public ITopLevelPass {
   public:
    using ITopLevelPass::ITopLevelPass;
    ~IRenderer() override = default;

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

    /// Iterate all child passes with a callback.
    template <typename Fn>
    void for_each_subpass(Fn&& fn) {
        for (auto& c : m_children) fn(*c);
    }

    // ── Lifecycle: auto-forwarded to all children ──

    /// Frame graph: delegates entirely to the renderer. The renderer calls
    /// child passes explicitly at the points it chooses.
    void add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) override {
        do_add_to_frame_graph(fg, ctx);
    }

    void on_shaders_reloaded(const webgpu::Device& device) override {
        for (auto& c : m_children) c->on_shaders_reloaded(device);
        ITopLevelPass::on_shaders_reloaded(device);
    }

    void draw_imgui() override;

    void draw_viewport_overlay(const ViewportOverlayParams& params) override {
        for (auto& c : m_children) c->draw_viewport_overlay(params);
    }

    void update_texture_refs(FrameGraph& fg) override {
        for (auto& c : m_children) c->update_texture_refs(fg);
    }

    ResourceHandle color_output() const {
        return m_color;
    }
    ResourceHandle depth_output() const {
        return m_depth;
    }

   protected:
    void do_setup(const webgpu::Device& device) override {
        for (auto& c : m_children) c->setup(device);
        do_renderer_setup(device);
    }

    virtual void do_renderer_setup(const webgpu::Device& device) = 0;
    virtual void do_add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) = 0;
    virtual void do_draw_imgui() {};

    ResourceHandle m_color;
    ResourceHandle m_depth;

   private:
    std::vector<std::unique_ptr<IPass>> m_children;
};

}  // namespace pts::rendering
