#pragma once

#include <core/rendering/graph.h>
#include <imgui.h>

namespace pts::rendering {

class IRhiBackend {
   public:
    virtual ~IRhiBackend() = default;

    virtual void new_frame() = 0;
    virtual void render(bool framebuffer_resized) = 0;
    virtual void resize_render_graph(uint32_t width, uint32_t height) = 0;
    virtual void set_render_graph_current() = 0;
    virtual void clear_render_graph_current() = 0;

    [[nodiscard]] virtual auto render_graph_api() const noexcept -> const PtsRenderGraphApi* = 0;
    [[nodiscard]] virtual auto output_texture() const noexcept -> PtsTexture = 0;
    [[nodiscard]] virtual auto output_imgui_id() const noexcept -> ImTextureID = 0;
};

}  // namespace pts::rendering
