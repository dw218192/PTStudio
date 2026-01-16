#pragma once

#include <core/rendering/graph.h>
#include <imgui.h>

#include <memory>

struct GLFWwindow;

namespace pts::rendering {
class RenderingHost {
   public:
    explicit RenderingHost(GLFWwindow* window);
    ~RenderingHost();

    RenderingHost(const RenderingHost&) = delete;
    RenderingHost& operator=(const RenderingHost&) = delete;
    RenderingHost(RenderingHost&&) = delete;
    RenderingHost& operator=(RenderingHost&&) = delete;

    void new_frame();
    void render(bool framebuffer_resized);
    void resize_render_graph(uint32_t width, uint32_t height);
    void set_render_graph_current();
    void clear_render_graph_current();

    [[nodiscard]] auto render_graph_api() const noexcept -> const PtsRenderGraphApi*;
    [[nodiscard]] auto output_texture() const noexcept -> PtsTexture;
    [[nodiscard]] auto output_imgui_id() const noexcept -> ImTextureID;

   private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
}  // namespace pts::rendering
