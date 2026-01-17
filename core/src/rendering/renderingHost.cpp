#include <core/rendering/renderingHost.h>

#include "rhiBackend.h"
#include "vulkanBackend.h"

namespace pts::rendering {
struct RenderingHost::Impl {
    explicit Impl(GLFWwindow* window, LoggingManager& logging_manager)
        : backend(std::make_unique<VulkanBackend>(window, logging_manager)) {
    }

    std::unique_ptr<IRhiBackend> backend;
};

RenderingHost::RenderingHost(GLFWwindow* window, LoggingManager& logging_manager)
    : m_impl(std::make_unique<Impl>(window, logging_manager)) {
}

RenderingHost::~RenderingHost() = default;

void RenderingHost::new_frame() {
    m_impl->backend->new_frame();
}

void RenderingHost::render(bool framebuffer_resized) {
    m_impl->backend->render(framebuffer_resized);
}

void RenderingHost::resize_render_graph(uint32_t width, uint32_t height) {
    m_impl->backend->resize_render_graph(width, height);
}

void RenderingHost::set_render_graph_current() {
    m_impl->backend->set_render_graph_current();
}

void RenderingHost::clear_render_graph_current() {
    m_impl->backend->clear_render_graph_current();
}

auto RenderingHost::render_graph_api() const noexcept -> const PtsRenderGraphApi* {
    return m_impl->backend->render_graph_api();
}

auto RenderingHost::output_texture() const noexcept -> PtsTexture {
    return m_impl->backend->output_texture();
}

auto RenderingHost::output_imgui_id() const noexcept -> ImTextureID {
    return m_impl->backend->output_imgui_id();
}
}  // namespace pts::rendering
