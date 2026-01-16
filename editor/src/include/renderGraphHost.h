#pragma once

#include <core/rendering/graph.h>

#include <imgui_impl_vulkan.h>
#include <vulkan/vulkan.hpp>

#include <vector>

namespace PTS::Editor {
class RenderGraphHost {
  public:
    RenderGraphHost(vk::PhysicalDevice physical_device, vk::Device device, vk::Queue queue,
                    uint32_t queue_family);
    ~RenderGraphHost();

    RenderGraphHost(const RenderGraphHost&) = delete;
    RenderGraphHost& operator=(const RenderGraphHost&) = delete;
    RenderGraphHost(RenderGraphHost&&) = delete;
    RenderGraphHost& operator=(RenderGraphHost&&) = delete;

    void resize(uint32_t width, uint32_t height);
    void set_current();
    void clear_current();

    [[nodiscard]] auto output_texture() const noexcept -> PtsTexture {
        return m_output_texture;
    }
    [[nodiscard]] auto output_imgui_id() const noexcept -> ImTextureID {
        return m_output_imgui_id;
    }
    [[nodiscard]] auto api() const noexcept -> const PtsRenderGraphApi* {
        return &m_api;
    }

  private:
    struct Pass {
        bool has_color{false};
        PtsAttachment color{};
    };
    struct Graph {
        std::vector<Pass> passes;
    };

    static thread_local RenderGraphHost* s_current;

    static PtsGraph begin_graph();
    static void end_graph(PtsGraph g);
    static PtsTexture import_texture(PtsGraph g, PtsTexture external_tex, const char* debug_name);
    static PtsTexView create_tex_view(PtsGraph g, const PtsTextureViewDesc* desc,
                                      const char* debug_name);
    static PtsPass add_pass(PtsGraph g, const PtsPassDesc* desc);
    static PtsGraphError get_last_error(PtsGraph g);
    static const char* get_error_message(PtsGraphError err);

    void execute(Graph& graph);
    void create_output_resources();
    void destroy_output_resources();
    void transition_image_layout(vk::CommandBuffer cmd_buf, vk::ImageLayout new_layout);
    void clear_color(vk::CommandBuffer cmd_buf, const float rgba[4]);
    [[nodiscard]] auto allocate_command_buffer() -> vk::CommandBuffer;
    [[nodiscard]] auto find_memory_type(uint32_t type_bits,
                                        vk::MemoryPropertyFlags flags) -> uint32_t;

    vk::PhysicalDevice m_physical_device{};
    vk::Device m_device{};
    vk::Queue m_queue{};
    uint32_t m_queue_family{0};

    vk::CommandPool m_command_pool{};

    vk::Image m_output_image{};
    vk::DeviceMemory m_output_memory{};
    vk::ImageView m_output_view{};
    vk::Sampler m_output_sampler{};
    vk::ImageLayout m_output_layout{vk::ImageLayout::eUndefined};
    vk::Extent2D m_output_extent{};
    vk::Format m_output_format{vk::Format::eR8G8B8A8Unorm};
    ImTextureID m_output_imgui_id{nullptr};
    PtsTexture m_output_texture{};

    PtsRenderGraphApi m_api{};
};
}  // namespace PTS::Editor

