#include "vulkanRenderGraph.h"

#include <array>
#include <cstring>
#include <utility>

namespace pts::rendering {
thread_local VulkanRenderGraph* VulkanRenderGraph::s_current = nullptr;

VulkanRenderGraph::VulkanRenderGraph(std::shared_ptr<VulkanRhi> rhi,
                                     pts::LoggingManager& logging_manager)
    : m_rhi_owner(std::move(rhi)), m_rhi(*m_rhi_owner) {
    m_logger = logging_manager.get_logger_shared("RenderGraph");
    m_logger->info("RenderGraph created");
    m_api.get_last_error = &VulkanRenderGraph::get_last_error;
    m_api.get_error_message = &VulkanRenderGraph::get_error_message;
    m_api.begin = &VulkanRenderGraph::begin_graph;
    m_api.end = &VulkanRenderGraph::end_graph;
    m_api.import_texture = &VulkanRenderGraph::import_texture;
    m_api.create_tex_view = &VulkanRenderGraph::create_tex_view;
    m_api.add_pass = &VulkanRenderGraph::add_pass;

    m_api.create_texture = nullptr;
    m_api.create_buffer = nullptr;
    m_api.create_buf_view = nullptr;
    m_api.import_buffer = nullptr;
    m_api.create_sampler = nullptr;
    m_api.create_graphics_pipeline = nullptr;
    m_api.create_compute_pipeline = nullptr;
    m_api.bb_set_u64 = nullptr;
    m_api.bb_get_u64 = nullptr;
    m_api.bb_set_tex = nullptr;
    m_api.bb_get_tex = nullptr;
    m_api.bb_set_buf = nullptr;
    m_api.bb_get_buf = nullptr;
    m_api.get_or_create_persistent_texture = nullptr;
    m_api.get_or_create_persistent_buffer = nullptr;

    auto pool_info = vk::CommandPoolCreateInfo{}
                         .setQueueFamilyIndex(m_rhi.queue_family())
                         .setFlags(vk::CommandPoolCreateFlagBits::eTransient);
    m_command_pool = m_rhi.device().createCommandPool(pool_info);
}

VulkanRenderGraph::~VulkanRenderGraph() {
    if (m_logger) {
        m_logger->info("RenderGraph destroyed");
    }
    destroy_output_resources();
    if (m_command_pool) {
        m_rhi.device().destroyCommandPool(m_command_pool);
    }
}

void VulkanRenderGraph::resize(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) {
        if (m_logger) {
            m_logger->debug("RenderGraph resize skipped for zero extent");
        }
        return;
    }
    if (m_output_extent.width == width && m_output_extent.height == height) {
        return;
    }
    if (m_logger) {
        m_logger->info("RenderGraph resized: {}x{}", width, height);
    }
    destroy_output_resources();
    m_output_extent = vk::Extent2D{width, height};
    create_output_resources();
}

void VulkanRenderGraph::set_current() {
    s_current = this;
}

void VulkanRenderGraph::clear_current() {
    if (s_current == this) {
        s_current = nullptr;
    }
}

PtsGraph VulkanRenderGraph::begin_graph() {
    if (!s_current) {
        return PtsGraph{};
    }
    auto* graph = new Graph();
    PtsGraph handle{};
    handle.h = reinterpret_cast<uint64_t>(graph);
    return handle;
}

void VulkanRenderGraph::end_graph(PtsGraph g) {
    if (!s_current || PTS_IS_NULL(g)) {
        return;
    }
    auto* graph = reinterpret_cast<Graph*>(g.h);
    s_current->execute(*graph);
    delete graph;
}

PtsTexture VulkanRenderGraph::import_texture(PtsGraph, PtsTexture external_tex, const char*) {
    return external_tex;
}

PtsTexView VulkanRenderGraph::create_tex_view(PtsGraph, const PtsTextureViewDesc* desc,
                                              const char*) {
    if (!s_current || !desc) {
        return PtsTexView{};
    }
    if (desc->tex.h == s_current->m_output_texture.h) {
        PtsTexView view{};
        view.h = desc->tex.h + 1;
        return view;
    }
    return PtsTexView{};
}

PtsPass VulkanRenderGraph::add_pass(PtsGraph g, const PtsPassDesc* desc) {
    if (!s_current || PTS_IS_NULL(g) || !desc) {
        return PtsPass{};
    }
    auto* graph = reinterpret_cast<Graph*>(g.h);
    Pass pass{};
    if (desc->color_attachments && desc->color_count > 0) {
        pass.has_color = true;
        pass.color = desc->color_attachments[0];
    }
    graph->passes.push_back(pass);
    PtsPass handle{};
    handle.h = static_cast<uint64_t>(graph->passes.size());
    return handle;
}

PtsGraphError VulkanRenderGraph::get_last_error(PtsGraph) {
    return PTS_GRAPH_OK;
}

const char* VulkanRenderGraph::get_error_message(PtsGraphError err) {
    switch (err) {
        case PTS_GRAPH_OK:
            return "ok";
        case PTS_GRAPH_ERR_INVALID_HANDLE:
            return "invalid handle";
        case PTS_GRAPH_ERR_INVALID_DESC:
            return "invalid desc";
        case PTS_GRAPH_ERR_OUT_OF_MEMORY:
            return "out of memory";
        case PTS_GRAPH_ERR_FORMAT_NOT_SUPPORTED:
            return "format not supported";
        case PTS_GRAPH_ERR_RESOURCE_LIMIT:
            return "resource limit";
        case PTS_GRAPH_ERR_INVALID_STATE:
            return "invalid state";
        case PTS_GRAPH_ERR_NOT_FOUND:
            return "not found";
        default:
            return "unknown";
    }
}

void VulkanRenderGraph::execute(Graph& graph) {
    if (!m_output_image) {
        return;
    }

    auto cmd_buf = allocate_command_buffer();
    cmd_buf.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    bool wrote_output = false;
    for (auto const& pass : graph.passes) {
        if (!pass.has_color) {
            continue;
        }
        if (pass.color.view.h != m_output_texture.h + 1) {
            continue;
        }
        transition_image_layout(cmd_buf, vk::ImageLayout::eTransferDstOptimal);
        clear_color(cmd_buf, pass.color.clear.rgba);
        transition_image_layout(cmd_buf, vk::ImageLayout::eShaderReadOnlyOptimal);
        wrote_output = true;
    }
    if (!wrote_output) {
        transition_image_layout(cmd_buf, vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    cmd_buf.end();
    m_rhi.queue().submit(vk::SubmitInfo{}.setCommandBuffers(cmd_buf), {});
    m_rhi.queue().waitIdle();
    m_rhi.device().freeCommandBuffers(m_command_pool, cmd_buf);
}

void VulkanRenderGraph::create_output_resources() {
    if (m_output_extent.width == 0 || m_output_extent.height == 0) {
        return;
    }
    if (m_logger) {
        m_logger->info("Creating render graph output resources: {}x{}", m_output_extent.width,
                       m_output_extent.height);
    }

    auto image_info =
        vk::ImageCreateInfo{}
            .setImageType(vk::ImageType::e2D)
            .setFormat(m_output_format)
            .setExtent({m_output_extent.width, m_output_extent.height, 1})
            .setMipLevels(1)
            .setArrayLayers(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setTiling(vk::ImageTiling::eOptimal)
            .setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled |
                      vk::ImageUsageFlagBits::eColorAttachment)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setInitialLayout(vk::ImageLayout::eUndefined);

    m_output_image = m_rhi.device().createImage(image_info);
    auto mem_req = m_rhi.device().getImageMemoryRequirements(m_output_image);
    auto mem_index =
        find_memory_type(mem_req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    auto alloc_info =
        vk::MemoryAllocateInfo{}.setAllocationSize(mem_req.size).setMemoryTypeIndex(mem_index);
    m_output_memory = m_rhi.device().allocateMemory(alloc_info);
    m_rhi.device().bindImageMemory(m_output_image, m_output_memory, 0);

    auto view_info = vk::ImageViewCreateInfo{}
                         .setImage(m_output_image)
                         .setViewType(vk::ImageViewType::e2D)
                         .setFormat(m_output_format)
                         .setSubresourceRange(vk::ImageSubresourceRange{}
                                                  .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                                  .setBaseMipLevel(0)
                                                  .setLevelCount(1)
                                                  .setBaseArrayLayer(0)
                                                  .setLayerCount(1));
    m_output_view = m_rhi.device().createImageView(view_info);

    auto sampler_info = vk::SamplerCreateInfo{}
                            .setMagFilter(vk::Filter::eLinear)
                            .setMinFilter(vk::Filter::eLinear)
                            .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
                            .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
                            .setAddressModeW(vk::SamplerAddressMode::eClampToEdge);
    m_output_sampler = m_rhi.device().createSampler(sampler_info);

    m_output_texture = PtsTexture{};
    m_output_texture.h = 1;
    m_output_layout = vk::ImageLayout::eUndefined;

    auto cmd_buf = allocate_command_buffer();
    cmd_buf.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    transition_image_layout(cmd_buf, vk::ImageLayout::eShaderReadOnlyOptimal);
    cmd_buf.end();
    m_rhi.queue().submit(vk::SubmitInfo{}.setCommandBuffers(cmd_buf), {});
    m_rhi.queue().waitIdle();
    m_rhi.device().freeCommandBuffers(m_command_pool, cmd_buf);
}

void VulkanRenderGraph::destroy_output_resources() {
    if (m_output_image && m_logger) {
        m_logger->info("Destroying render graph output resources");
    }
    if (m_output_sampler) {
        m_rhi.device().destroySampler(m_output_sampler);
        m_output_sampler = VK_NULL_HANDLE;
    }
    if (m_output_view) {
        m_rhi.device().destroyImageView(m_output_view);
        m_output_view = VK_NULL_HANDLE;
    }
    if (m_output_image) {
        m_rhi.device().destroyImage(m_output_image);
        m_output_image = VK_NULL_HANDLE;
    }
    if (m_output_memory) {
        m_rhi.device().freeMemory(m_output_memory);
        m_output_memory = VK_NULL_HANDLE;
    }
}

void VulkanRenderGraph::transition_image_layout(vk::CommandBuffer cmd_buf,
                                                vk::ImageLayout new_layout) {
    if (m_output_layout == new_layout) {
        return;
    }

    auto barrier = vk::ImageMemoryBarrier{}
                       .setOldLayout(m_output_layout)
                       .setNewLayout(new_layout)
                       .setImage(m_output_image)
                       .setSubresourceRange(vk::ImageSubresourceRange{}
                                                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                                .setBaseMipLevel(0)
                                                .setLevelCount(1)
                                                .setBaseArrayLayer(0)
                                                .setLayerCount(1));

    vk::PipelineStageFlags src_stage = vk::PipelineStageFlagBits::eTopOfPipe;
    vk::PipelineStageFlags dst_stage = vk::PipelineStageFlagBits::eTransfer;
    if (m_output_layout == vk::ImageLayout::eTransferDstOptimal &&
        new_layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        src_stage = vk::PipelineStageFlagBits::eTransfer;
        dst_stage = vk::PipelineStageFlagBits::eFragmentShader;
    } else if (m_output_layout == vk::ImageLayout::eUndefined &&
               new_layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.setSrcAccessMask({}).setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        src_stage = vk::PipelineStageFlagBits::eTopOfPipe;
        dst_stage = vk::PipelineStageFlagBits::eFragmentShader;
    } else if (new_layout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.setSrcAccessMask({}).setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
    }

    cmd_buf.pipelineBarrier(src_stage, dst_stage, {}, nullptr, nullptr, barrier);
    m_output_layout = new_layout;
}

void VulkanRenderGraph::clear_color(vk::CommandBuffer cmd_buf, const float rgba[4]) {
    auto color = vk::ClearColorValue{std::array<float, 4>{rgba[0], rgba[1], rgba[2], rgba[3]}};
    auto range = vk::ImageSubresourceRange{}
                     .setAspectMask(vk::ImageAspectFlagBits::eColor)
                     .setBaseMipLevel(0)
                     .setLevelCount(1)
                     .setBaseArrayLayer(0)
                     .setLayerCount(1);
    cmd_buf.clearColorImage(m_output_image, m_output_layout, color, range);
}

vk::CommandBuffer VulkanRenderGraph::allocate_command_buffer() {
    auto alloc_info = vk::CommandBufferAllocateInfo{}
                          .setCommandPool(m_command_pool)
                          .setLevel(vk::CommandBufferLevel::ePrimary)
                          .setCommandBufferCount(1);
    auto buffers = m_rhi.device().allocateCommandBuffers(alloc_info);
    return buffers.front();
}

uint32_t VulkanRenderGraph::find_memory_type(uint32_t type_bits, vk::MemoryPropertyFlags flags) {
    auto props = m_rhi.physical_device().getMemoryProperties();
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return 0;
}
}  // namespace pts::rendering
