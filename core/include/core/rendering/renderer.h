#pragma once

#include <core/rendering/renderPass.h>

#include <memory>
#include <vector>

namespace pts::rendering {

class IRenderer : public IRenderPass {
   public:
    using IRenderPass::IRenderPass;
    ~IRenderer() override = default;

    /// Add a child pass that runs BEFORE the main pass (e.g. shadows, G-buffer).
    template <typename T, typename... Args>
    T& add_pass(Args&&... args) {
        auto p = std::make_unique<T>(std::forward<Args>(args)...);
        auto& ref = *p;
        m_children.push_back(std::move(p));
        return ref;
    }

    /// Add a child pass that runs AFTER the main pass (e.g. SSAO, SSR).
    template <typename T, typename... Args>
    T& add_post_pass(Args&&... args) {
        auto p = std::make_unique<T>(std::forward<Args>(args)...);
        auto& ref = *p;
        m_post_children.push_back(std::move(p));
        return ref;
    }

    /// Iterate all child passes (pre and post) with a callback.
    template <typename Fn>
    void for_each_subpass(Fn&& fn) {
        for (auto& c : m_children) fn(*c);
        for (auto& c : m_post_children) fn(*c);
    }

    /// Find a child pass by type (searches both pre and post children).
    template <typename T>
    T* get_pass() const {
        for (auto& c : m_children) {
            if (auto* p = dynamic_cast<T*>(c.get())) return p;
        }
        for (auto& c : m_post_children) {
            if (auto* p = dynamic_cast<T*>(c.get())) return p;
        }
        return nullptr;
    }

    // Lifecycle overrides — forward to children, then self, then post-children
    void add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) override {
        for (auto& c : m_children) {
            if (c->is_ready()) c->add_to_frame_graph(fg, ctx);
        }
        do_add_to_frame_graph(fg, ctx);
        for (auto& c : m_post_children) {
            if (c->is_ready()) c->add_to_frame_graph(fg, ctx);
        }
    }

    void on_shaders_reloaded(const webgpu::Device& device) override {
        for (auto& c : m_children) c->on_shaders_reloaded(device);
        for (auto& c : m_post_children) c->on_shaders_reloaded(device);
        IRenderPass::on_shaders_reloaded(device);
    }

    void draw_imgui() override;

    void draw_viewport_overlay(const ViewportOverlayParams& params) override {
        for (auto& c : m_children) c->draw_viewport_overlay(params);
        for (auto& c : m_post_children) c->draw_viewport_overlay(params);
    }

    void update_texture_refs(FrameGraph& fg) override {
        for (auto& c : m_children) c->update_texture_refs(fg);
        for (auto& c : m_post_children) c->update_texture_refs(fg);
    }

   protected:
    // do_setup forwards to all children, then calls do_renderer_setup
    void do_setup(const webgpu::Device& device) override {
        for (auto& c : m_children) c->setup(device);
        do_renderer_setup(device);
        for (auto& c : m_post_children) c->setup(device);
    }

    virtual void do_renderer_setup(const webgpu::Device& device) = 0;
    virtual void do_add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) = 0;
    virtual void do_draw_imgui() {};

   private:
    std::vector<std::unique_ptr<IRenderPass>> m_children;       // run before main pass
    std::vector<std::unique_ptr<IRenderPass>> m_post_children;  // run after main pass
};

}  // namespace pts::rendering
