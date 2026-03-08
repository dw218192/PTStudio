#pragma once

#include <core/rendering/webgpu/webgpu.h>

#include <boost/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace spdlog {
class logger;
}

namespace pts::webgpu {
class Device;
}

namespace pts::rendering {

struct ResourceHandle {
    uint32_t index = UINT32_MAX;
    [[nodiscard]] bool is_valid() const {
        return index != UINT32_MAX;
    }
};

struct TextureDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    WGPUTextureFormat format = WGPUTextureFormat_BGRA8Unorm;
    WGPUTextureUsage usage = WGPUTextureUsage_RenderAttachment;
    WGPUColor clear_color = {0, 0, 0, 1};
    float depth_clear_value = 1.0f;
};

namespace detail {

struct CachedTexture : boost::intrusive_ref_counter<CachedTexture, boost::thread_unsafe_counter> {
    WGPUTexture texture = nullptr;
    WGPUTextureView view = nullptr;
    TextureDesc desc;
    bool used_this_frame = false;

    ~CachedTexture();
    CachedTexture() = default;
    CachedTexture(const CachedTexture&) = delete;
    CachedTexture& operator=(const CachedTexture&) = delete;
};

}  // namespace detail

class TextureRef {
   public:
    WGPUTextureView view() const {
        return m_cached ? m_cached->view : m_imported_view;
    }
    explicit operator bool() const {
        return m_cached || m_imported_view;
    }

   private:
    friend class FrameGraph;
    boost::intrusive_ptr<detail::CachedTexture> m_cached;
    WGPUTextureView m_imported_view = nullptr;
};

using ExecuteFn = std::function<void(WGPURenderPassEncoder)>;

class PassBuilder {
   public:
    PassBuilder& color(ResourceHandle h);
    PassBuilder& depth(ResourceHandle h);
    PassBuilder& depth_readonly(ResourceHandle h);
    PassBuilder& present(ResourceHandle h);
    PassBuilder& read(ResourceHandle h);
    void execute(ExecuteFn fn);

   private:
    friend class FrameGraph;
    explicit PassBuilder(class FrameGraph& graph, uint32_t pass_index);

    FrameGraph& m_graph;
    uint32_t m_pass_index;
};

class FrameGraph {
   public:
    explicit FrameGraph(const webgpu::Device& device, std::shared_ptr<spdlog::logger> logger);
    ~FrameGraph();

    FrameGraph(const FrameGraph&) = delete;
    FrameGraph& operator=(const FrameGraph&) = delete;

    ResourceHandle import(std::string name, WGPUTextureView view, TextureDesc desc);
    ResourceHandle create(std::string name, TextureDesc desc);

    PassBuilder add_pass(std::string name);

    void begin_frame();
    void compile();
    void execute(WGPUCommandEncoder encoder);

    [[nodiscard]] TextureRef get_texture_ref(ResourceHandle h) const;

   private:
    friend class PassBuilder;

    [[nodiscard]] WGPUTextureView resolve_view(ResourceHandle h) const;

    struct Resource {
        std::string name;
        TextureDesc desc;
        WGPUTextureView imported_view = nullptr;  // non-null for imported resources
        uint32_t first_writer = UINT32_MAX;
        bool is_present = false;
    };

    struct ColorAttachmentInfo {
        ResourceHandle handle;
        bool is_read = false;
        bool is_write = false;
    };

    struct DepthAttachmentInfo {
        ResourceHandle handle;
        bool is_read = false;
        bool is_write = false;
    };

    struct Pass {
        std::string name;
        uint32_t index = 0;
        std::vector<ColorAttachmentInfo> color_attachments;
        DepthAttachmentInfo depth_attachment;
        bool has_depth = false;
        std::vector<ResourceHandle> reads;
        ExecuteFn execute_fn;

        // Derived during compile
        WGPULoadOp color_load_op = WGPULoadOp_Clear;
        WGPUStoreOp color_store_op = WGPUStoreOp_Store;
        WGPULoadOp depth_load_op = WGPULoadOp_Clear;
        WGPUStoreOp depth_store_op = WGPUStoreOp_Store;
        bool depth_read_only = false;
    };

    void allocate_textures();
    void evict_unused();

    const webgpu::Device& m_device;
    std::shared_ptr<spdlog::logger> m_logger;

    std::vector<Resource> m_resources;
    std::vector<Pass> m_passes;
    std::unordered_map<std::string, boost::intrusive_ptr<detail::CachedTexture>> m_texture_cache;
};

}  // namespace pts::rendering
