#pragma once

#include <core/rendering/graph.h>
#include <core/rendering/windowing.h>
#include <imgui.h>

#include <memory>

namespace pts {
class LoggingManager;
}

namespace pts::rendering {
class Rendering {
   public:
    explicit Rendering(IWindowing& windowing, pts::LoggingManager& logging_manager);
    ~Rendering();

    Rendering(const Rendering&) = delete;
    Rendering& operator=(const Rendering&) = delete;
    Rendering(Rendering&&) = delete;
    Rendering& operator=(Rendering&&) = delete;

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
