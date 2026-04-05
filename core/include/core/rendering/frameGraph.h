#pragma once

#include <core/diagnostics.h>
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

class IPass;

struct TextureHandle {
    uint32_t index = UINT32_MAX;
    [[nodiscard]] bool is_valid() const {
        return index != UINT32_MAX;
    }
};

// Backward-compatible alias
using ResourceHandle = TextureHandle;

struct TextureDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t array_layers = 1;
    WGPUTextureFormat format = WGPUTextureFormat_BGRA8Unorm;
    WGPUTextureUsage usage = WGPUTextureUsage_RenderAttachment;
    WGPUColor clear_color = {0, 0, 0, 1};
    float depth_clear_value = 1.0f;
    bool force_array_view = false;  // create 2DArray view even with 1 layer
};

struct BufferDesc {
    uint64_t size = 0;
    WGPUBufferUsage usage = WGPUBufferUsage_None;
};

struct BufferHandle {
    uint32_t index = UINT32_MAX;
    [[nodiscard]] bool is_valid() const {
        return index != UINT32_MAX;
    }
};

struct BindGroupEntry {
    uint32_t binding = 0;
    // Exactly one of these is set per entry:
    BufferHandle buffer;
    uint64_t buffer_offset = 0;
    uint64_t buffer_size = 0;  // 0 = whole buffer
    TextureHandle texture;
    uint32_t texture_layer = UINT32_MAX;  // specific layer if != UINT32_MAX (for ticket 4)
    WGPUSampler sampler = nullptr;
    WGPUTextureView external_view = nullptr;
    WGPUBuffer external_buffer = nullptr;
    uint64_t external_buffer_size = 0;
};

struct BindGroupDesc {
    WGPUBindGroupLayout layout = nullptr;
    std::vector<BindGroupEntry> entries;
};

struct BindGroupHandle {
    uint32_t index = UINT32_MAX;
    [[nodiscard]] bool is_valid() const {
        return index != UINT32_MAX;
    }
};

namespace detail {

template <typename Derived>
struct CachedResource : boost::intrusive_ref_counter<Derived, boost::thread_unsafe_counter> {
    bool used_this_frame = false;
    uint64_t version = 0;
};

struct CachedTexture : CachedResource<CachedTexture> {
    WGPUTexture texture = nullptr;
    WGPUTextureView view = nullptr;
    std::vector<WGPUTextureView> layer_views;
    TextureDesc desc;

    ~CachedTexture();
    CachedTexture() = default;
    CachedTexture(const CachedTexture&) = delete;
    CachedTexture& operator=(const CachedTexture&) = delete;
};

struct CachedBuffer : CachedResource<CachedBuffer> {
    WGPUBuffer buffer = nullptr;
    BufferDesc desc;
    bool owned = true;

    ~CachedBuffer();
    CachedBuffer() = default;
    CachedBuffer(const CachedBuffer&) = delete;
    CachedBuffer& operator=(const CachedBuffer&) = delete;
};

struct CachedBindGroup : CachedResource<CachedBindGroup> {
    WGPUBindGroup bind_group = nullptr;
    std::vector<uint64_t> input_versions_snapshot;

    ~CachedBindGroup();
    CachedBindGroup() = default;
    CachedBindGroup(const CachedBindGroup&) = delete;
    CachedBindGroup& operator=(const CachedBindGroup&) = delete;
};

}  // namespace detail

template <typename CachedType>
class ResourceRef {
   public:
    explicit operator bool() const {
        return m_cached != nullptr;
    }

   protected:
    friend class FrameGraph;
    boost::intrusive_ptr<CachedType> m_cached;
};

class TextureRef : public ResourceRef<detail::CachedTexture> {
   public:
    WGPUTextureView view() const {
        return m_cached ? m_cached->view : nullptr;
    }
    WGPUTexture texture() const {
        return m_cached ? m_cached->texture : nullptr;
    }
    WGPUTextureView layer_view(uint32_t i) const {
        PRECONDITION(m_cached != nullptr);
        PRECONDITION(i < m_cached->layer_views.size());
        return m_cached->layer_views[i];
    }
    uint32_t layer_count() const {
        return m_cached ? static_cast<uint32_t>(m_cached->layer_views.size()) : 0;
    }
};

class BufferRef : public ResourceRef<detail::CachedBuffer> {
   public:
    WGPUBuffer handle() const {
        return m_cached ? m_cached->buffer : nullptr;
    }
    std::size_t size() const {
        return m_cached ? m_cached->desc.size : 0;
    }
};

class BindGroupRef : public ResourceRef<detail::CachedBindGroup> {
   public:
    WGPUBindGroup handle() const {
        return m_cached ? m_cached->bind_group : nullptr;
    }
};

enum class PassType { Render, Compute };

using ExecuteRenderFn = std::function<void(WGPURenderPassEncoder)>;
using ExecuteComputeFn = std::function<void(WGPUComputePassEncoder)>;

// Keep backward-compatible alias
using ExecuteFn = ExecuteRenderFn;

class PassBuilder {
   public:
    PassBuilder& color(ResourceHandle h);
    PassBuilder& color(ResourceHandle h, uint32_t layer);
    PassBuilder& color(WGPUTextureView view, WGPUColor clear_color = {});
    PassBuilder& depth(ResourceHandle h);
    PassBuilder& depth(ResourceHandle h, uint32_t layer);
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

    // --- String-based API (used in tests and for top-level resources) ---
    ResourceHandle create(std::string name, TextureDesc desc);
    ResourceHandle find_or_create(std::string name, TextureDesc desc);
    [[nodiscard]] std::optional<ResourceHandle> find(const std::string& name) const;

    BufferHandle find_or_create_buffer(std::string name, BufferDesc desc);
    BufferHandle import_buffer(std::string name, WGPUBuffer buf, std::size_t size);
    [[nodiscard]] std::optional<BufferHandle> find_buffer(const std::string& name) const;

    BindGroupHandle find_or_create_bind_group(std::string name, BindGroupDesc desc);
    [[nodiscard]] std::optional<BindGroupHandle> find_bind_group(const std::string& name) const;
    [[nodiscard]] BindGroupRef get_bind_group_ref(BindGroupHandle h) const;

    // --- Pass-based API (auto-namespaced by pass name) ---
    ResourceHandle find_or_create(const IPass* pass, TextureDesc desc, const char* label = nullptr);
    BufferHandle find_or_create_buffer(const IPass* pass, BufferDesc desc,
                                       const char* label = nullptr);
    BufferHandle import_buffer(const IPass* pass, WGPUBuffer buf, std::size_t size,
                               const char* label = nullptr);
    BindGroupHandle find_or_create_bind_group(const IPass* pass, BindGroupDesc desc,
                                              const char* label = nullptr);

    PassBuilder add_pass(std::string name);

    void begin_frame();
    void compile();
    void execute(WGPUCommandEncoder encoder);

    [[nodiscard]] TextureRef get_texture_ref(ResourceHandle h) const;
    [[nodiscard]] BufferRef get_buffer_ref(BufferHandle h) const;
    [[nodiscard]] size_t cached_texture_count() const {
        return m_texture_cache.size();
    }
    [[nodiscard]] size_t cached_buffer_count() const {
        return m_buffer_cache.size();
    }
    [[nodiscard]] size_t cached_bind_group_count() const {
        return m_bg_cache.size();
    }

   private:
    friend class PassBuilder;

    [[nodiscard]] WGPUTextureView resolve_view(ResourceHandle h) const;
    [[nodiscard]] WGPUTextureView resolve_layer_view(ResourceHandle h, uint32_t layer) const;

    struct Resource {
        std::string name;
        TextureDesc desc;
        WGPUTextureView external_view = nullptr;
        uint32_t first_writer = UINT32_MAX;
    };

    struct ColorAttachmentInfo {
        ResourceHandle handle;
        uint32_t layer = UINT32_MAX;
        bool is_read = false;
        bool is_write = false;

        // Derived during compile (per-attachment load/store ops for MRT)
        WGPULoadOp load_op = WGPULoadOp_Clear;
        WGPUStoreOp store_op = WGPUStoreOp_Store;
    };

    struct DepthAttachmentInfo {
        ResourceHandle handle;
        uint32_t layer = UINT32_MAX;
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
    void allocate_buffers();
    void allocate_bind_groups();
    void evict_unused();

    const webgpu::Device& m_device;
    std::shared_ptr<spdlog::logger> m_logger;

    std::vector<Resource> m_resources;
    std::vector<Pass> m_passes;
    std::unordered_map<std::string, boost::intrusive_ptr<detail::CachedTexture>> m_texture_cache;

    struct BufferResource {
        std::string name;
        BufferDesc desc;
        WGPUBuffer external_buffer = nullptr;
        std::size_t external_size = 0;
    };
    std::vector<BufferResource> m_buffer_resources;
    std::unordered_map<std::string, boost::intrusive_ptr<detail::CachedBuffer>> m_buffer_cache;

    struct BindGroupResource {
        std::string name;
        BindGroupDesc desc;
    };
    std::vector<BindGroupResource> m_bg_resources;
    std::unordered_map<std::string, boost::intrusive_ptr<detail::CachedBindGroup>> m_bg_cache;

    // Per-pass auto-naming counters, reset each begin_frame()
    struct PassCounters {
        uint32_t texture = 0;
        uint32_t buffer = 0;
        uint32_t bind_group = 0;
    };
    std::unordered_map<std::string_view, PassCounters> m_pass_counters;

    std::string make_pass_key(const IPass* pass, const char* label, const char* type_prefix);
};

}  // namespace pts::rendering
