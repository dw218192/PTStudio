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
        return m_cached ? m_cached->view : nullptr;
    }
    explicit operator bool() const {
        return m_cached != nullptr;
    }

   private:
    friend class FrameGraph;
    boost::intrusive_ptr<detail::CachedTexture> m_cached;
};

enum class PassType { Render, Compute };

using ExecuteRenderFn = std::function<void(WGPURenderPassEncoder)>;
using ExecuteComputeFn = std::function<void(WGPUComputePassEncoder)>;

// Keep backward-compatible alias
using ExecuteFn = ExecuteRenderFn;

class PassBuilder {
   public:
    PassBuilder& color(ResourceHandle h);
    PassBuilder& color(WGPUTextureView view, WGPUColor clear_color = {});
    PassBuilder& depth(ResourceHandle h);
    PassBuilder& depth(WGPUTextureView view, float clear_value = 1.0f);
    PassBuilder& depth_readonly(ResourceHandle h);
    PassBuilder& present();
    PassBuilder& read(ResourceHandle h);
    PassBuilder& storage_write(ResourceHandle h);
    void execute(ExecuteRenderFn fn);
    void execute(ExecuteComputeFn fn);

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

    ResourceHandle create(std::string name, TextureDesc desc);
    ResourceHandle find_or_create(std::string name, TextureDesc desc);

    PassBuilder add_pass(std::string name);

    void begin_frame();
    void compile();
    void execute(WGPUCommandEncoder encoder);

    [[nodiscard]] TextureRef get_texture_ref(ResourceHandle h) const;
    [[nodiscard]] size_t cached_texture_count() const {
        return m_texture_cache.size();
    }

   private:
    friend class PassBuilder;

    [[nodiscard]] WGPUTextureView resolve_view(ResourceHandle h) const;

    struct Resource {
        std::string name;
        TextureDesc desc;
        WGPUTextureView external_view = nullptr;
        uint32_t first_writer = UINT32_MAX;
    };

    struct ColorAttachmentInfo {
        ResourceHandle handle;
        bool is_read = false;
        bool is_write = false;

        // Derived during compile (per-attachment load/store ops for MRT)
        WGPULoadOp load_op = WGPULoadOp_Clear;
        WGPUStoreOp store_op = WGPUStoreOp_Store;
    };

    struct DepthAttachmentInfo {
        ResourceHandle handle;
        bool is_read = false;
        bool is_write = false;
    };

    struct Pass {
        std::string name;
        uint32_t index = 0;
        PassType type = PassType::Render;
        std::vector<ColorAttachmentInfo> color_attachments;
        DepthAttachmentInfo depth_attachment;
        bool has_depth = false;
        bool is_present = false;
        std::vector<ResourceHandle> reads;
        ExecuteRenderFn render_fn;
        ExecuteComputeFn compute_fn;

        // Derived during compile
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
