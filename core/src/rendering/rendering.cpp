#include <core/rendering/rendering.h>

#include <stdexcept>

#include "imguiBackend.h"
#include "present.h"
#include "renderGraph.h"
#include "vulkan/vulkanPresent.h"
#include "vulkan/vulkanRhi.h"

namespace pts::rendering {
namespace {
#if PTS_GAPI_VULKAN
using RhiType = VulkanRhi;
using PresentType = VulkanPresent;
#else
#error "No GAPI selected."
#endif
}  // namespace

struct Rendering::Impl {
    explicit Impl(IWindowing& windowing, LoggingManager& logging_manager)
        : rhi(std::make_unique<RhiType>(windowing.required_vulkan_instance_extensions(),
                                        logging_manager)),
          present(create_present(windowing, logging_manager)),
          render_graph(std::make_unique<RenderGraph>(*vulkan_rhi, logging_manager)),
          imgui(std::make_unique<ImGuiBackend>(windowing, *present, *vulkan_rhi, logging_manager)) {
        auto const extent = present->framebuffer_extent();
        resize_render_graph(extent.width, extent.height);
    }

    ~Impl() {
        if (output_id) {
            vulkan_rhi->device().waitIdle();
            imgui->unregister_texture(output_id);
            output_id = nullptr;
        }
    }

    void resize_render_graph(uint32_t width, uint32_t height) {
        if (output_id) {
            vulkan_rhi->device().waitIdle();
            imgui->unregister_texture(output_id);
            output_id = nullptr;
        }
        if (width == 0 || height == 0) {
            return;
        }
        render_graph->resize(width, height);
        output_id = imgui->register_texture(render_graph->output_sampler(),
                                            render_graph->output_image_view(),
                                            render_graph->output_layout());
    }

    auto create_present(IWindowing& windowing, LoggingManager& logging_manager)
        -> std::unique_ptr<IPresent> {
        vulkan_rhi = dynamic_cast<VulkanRhi*>(rhi.get());
        if (!vulkan_rhi) {
            throw std::runtime_error("Vulkan RHI required for Vulkan present");
        }
        return std::make_unique<PresentType>(windowing, *vulkan_rhi, logging_manager);
    }

    std::unique_ptr<IRhi> rhi;
    VulkanRhi* vulkan_rhi{nullptr};
    std::unique_ptr<IPresent> present;
    std::unique_ptr<RenderGraph> render_graph;
    std::unique_ptr<ImGuiBackend> imgui;
    ImTextureID output_id{nullptr};
};

Rendering::Rendering(IWindowing& windowing, LoggingManager& logging_manager)
    : m_impl(std::make_unique<Impl>(windowing, logging_manager)) {
}

Rendering::~Rendering() = default;

void Rendering::new_frame() {
    m_impl->imgui->new_frame();
}

void Rendering::render(bool framebuffer_resized) {
    m_impl->imgui->render(framebuffer_resized);
}

void Rendering::resize_render_graph(uint32_t width, uint32_t height) {
    m_impl->resize_render_graph(width, height);
}

void Rendering::set_render_graph_current() {
    m_impl->render_graph->set_current();
}

void Rendering::clear_render_graph_current() {
    m_impl->render_graph->clear_current();
}

auto Rendering::render_graph_api() const noexcept -> const PtsRenderGraphApi* {
    return m_impl->render_graph ? m_impl->render_graph->api() : nullptr;
}

auto Rendering::output_texture() const noexcept -> PtsTexture {
    return m_impl->render_graph ? m_impl->render_graph->output_texture() : PtsTexture{};
}

auto Rendering::output_imgui_id() const noexcept -> ImTextureID {
    return m_impl->output_id;
}
}  // namespace pts::rendering
