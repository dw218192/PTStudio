#pragma once

#include <core/container/depTrackedSlotMap.h>
#include <core/defines.h>
#include <core/diagnostics.h>
#include <core/rendering/webgpu/webgpu.h>

#include <boost/container_hash/hash.hpp>
#include <boost/core/span.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

namespace pts::rendering {
class FallbackPool;
class IShaderCompiler;
}  // namespace pts::rendering

namespace spdlog {
class logger;
}

namespace pts::webgpu {
class Device;
}

namespace pts::rendering {

class IPass;
class FrameGraph;
class ExecuteContext;

// --------------------------------------------------------------------------
// Transparent string hasher / equal for heterogeneous lookup into
// string-keyed caches (find by string_view without allocating std::string).
// --------------------------------------------------------------------------

struct StringViewHash {
    using is_transparent = void;
    size_t operator()(std::string_view sv) const noexcept {
        return boost::hash<std::string_view>{}(sv);
    }
    size_t operator()(const std::string& s) const noexcept {
        return boost::hash<std::string_view>{}(s);
    }
    size_t operator()(const char* s) const noexcept {
        return boost::hash<std::string_view>{}(std::string_view{s});
    }
};

struct StringViewEqual {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const noexcept {
        return a == b;
    }
};

template <typename T>
using FlatStringMap = boost::unordered_flat_map<std::string, T, StringViewHash, StringViewEqual>;

enum class Lifetime { Frame, Persistent };

// --------------------------------------------------------------------------
// Handle types -- strong-typedef uint32_t, UINT32_MAX sentinel means invalid.
// --------------------------------------------------------------------------

struct TextureDeclHandle {
    uint32_t value = UINT32_MAX;
    explicit operator bool() const noexcept {
        return value != UINT32_MAX;
    }
    bool operator==(TextureDeclHandle o) const noexcept {
        return value == o.value;
    }
    bool operator!=(TextureDeclHandle o) const noexcept {
        return value != o.value;
    }
};

struct BufferDeclHandle {
    uint32_t value = UINT32_MAX;
    explicit operator bool() const noexcept {
        return value != UINT32_MAX;
    }
    bool operator==(BufferDeclHandle o) const noexcept {
        return value == o.value;
    }
    bool operator!=(BufferDeclHandle o) const noexcept {
        return value != o.value;
    }
};

struct DescriptorDeclHandle {
    uint32_t value = UINT32_MAX;
    explicit operator bool() const noexcept {
        return value != UINT32_MAX;
    }
    bool operator==(DescriptorDeclHandle o) const noexcept {
        return value == o.value;
    }
    bool operator!=(DescriptorDeclHandle o) const noexcept {
        return value != o.value;
    }
};

// --------------------------------------------------------------------------
// Desc types -- pure-data descriptions, no GPU handles.
// --------------------------------------------------------------------------

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

// --------------------------------------------------------------------------
// Compiled-phase types -- GPU handles valid, only reachable in execute lambdas
// via ExecuteContext::get(handle). FrameGraph owns them.
// --------------------------------------------------------------------------

struct Texture {
    WGPUTexture texture = nullptr;
    WGPUTextureView view = nullptr;
    std::vector<WGPUTextureView> layer_views;
    TextureDesc desc{};
    uint64_t version = 0;  // bumped when recreated

    Texture() = default;
    ~Texture();
    NO_COPY_MOVE(Texture);
};

struct Buffer {
    WGPUBuffer buffer = nullptr;
    uint64_t size = 0;
    WGPUBufferUsage usage = WGPUBufferUsage_None;
    uint64_t version = 0;
    bool owned = true;  // false for import_buffer

    Buffer() = default;
    ~Buffer();
    NO_COPY_MOVE(Buffer);
};

struct Descriptor {
    WGPUBindGroup bind_group = nullptr;
    uint64_t version = 0;

    Descriptor() = default;
    ~Descriptor();
    NO_COPY_MOVE(Descriptor);
};

// --------------------------------------------------------------------------
// Declaration-phase types -- no GPU handle fields. Stored in dense vectors
// indexed by handle. Back-pointer to compiled struct set by compile().
// --------------------------------------------------------------------------

struct TextureDecl {
    std::string debug_label;
    TextureDesc desc{};
    Lifetime lifetime = Lifetime::Frame;
    bool active = false;

    // Scheduling state (reset each frame for Frame lifetime)
    uint32_t first_writer = UINT32_MAX;
    uint32_t last_reader = UINT32_MAX;

    // Set by compile() when the handle appears in a pass's declarations.
    uint64_t last_active_frame = 0;

    // External view (if this decl wraps an externally-owned view like the
    // swapchain surface). When non-null, compile() does not allocate a
    // Texture -- ExecuteContext::get() is not expected to be used on these.
    WGPUTextureView external_view = nullptr;

    // Persistent initial upload (used by static-texture overload)
    const void* upload_data = nullptr;
    uint64_t upload_size = 0;
    uint32_t upload_bytes_per_row = 0;
    WGPUTextureDescriptor upload_desc{};
    WGPUTextureViewDimension upload_view_dim = WGPUTextureViewDimension_2D;
    bool has_upload = false;

    // Back-link to compiled result -- set by compile(), consumed by ExecuteContext.
    Texture* compiled = nullptr;

    TextureDecl() = default;
    NO_COPY(TextureDecl);
    TextureDecl(TextureDecl&&) noexcept = default;
    TextureDecl& operator=(TextureDecl&&) noexcept = default;
};

struct BufferDecl {
    std::string debug_label;
    BufferDesc desc{};
    Lifetime lifetime = Lifetime::Frame;
    bool active = false;

    uint32_t first_writer = UINT32_MAX;
    uint32_t last_reader = UINT32_MAX;
    uint64_t last_active_frame = 0;

    // External buffer (import_buffer). When set, compile() wraps it.
    WGPUBuffer external_buffer = nullptr;
    uint64_t external_size = 0;
    // Caller-provided version for imported buffers. Propagates into the
    // compiled Buffer's `version` so descriptors binding this buffer rebuild
    // when the external source (e.g. RenderWorld) mutates.
    uint64_t external_version = 0;

    // Persistent initial upload
    const void* upload_data = nullptr;
    uint64_t upload_size = 0;
    bool has_upload = false;

    Buffer* compiled = nullptr;

    BufferDecl() = default;
    NO_COPY(BufferDecl);
    BufferDecl(BufferDecl&&) noexcept = default;
    BufferDecl& operator=(BufferDecl&&) noexcept = default;
};

// --------------------------------------------------------------------------
// Descriptor entry variants -- managed bindings reference handles.
// --------------------------------------------------------------------------

struct ManagedBufferBinding {
    BufferDeclHandle handle;
    uint64_t offset = 0;
    uint64_t size = 0;  // 0 = whole buffer
};

struct ManagedTextureBinding {
    TextureDeclHandle handle;
    uint32_t layer = UINT32_MAX;
};

struct ExternalViewBinding {
    WGPUTextureView view = nullptr;
};

struct ExternalBufferBinding {
    WGPUBuffer buffer = nullptr;
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct SamplerBinding {
    WGPUSampler sampler = nullptr;
};

using BindingResource = std::variant<ManagedBufferBinding, ManagedTextureBinding,
                                     ExternalViewBinding, ExternalBufferBinding, SamplerBinding>;

struct DescriptorEntry {
    uint32_t binding = 0;
    BindingResource resource;
};

struct DescriptorDecl {
    std::string debug_label;
    WGPUBindGroupLayout layout = nullptr;
    std::vector<DescriptorEntry> entries;
    bool active = false;
    uint64_t last_active_frame = 0;

    Descriptor* compiled = nullptr;

    DescriptorDecl() = default;
    NO_COPY(DescriptorDecl);
    DescriptorDecl(DescriptorDecl&&) noexcept = default;
    DescriptorDecl& operator=(DescriptorDecl&&) noexcept = default;
};

// --------------------------------------------------------------------------
// ExecuteContext -- passed to pass execute lambdas. Provides O(1) accessor
// to compiled resources via handle indexing.
// --------------------------------------------------------------------------

class ExecuteContext {
   public:
    [[nodiscard]] const Texture& get(TextureDeclHandle h) const;
    [[nodiscard]] const Buffer& get(BufferDeclHandle h) const;
    [[nodiscard]] const Descriptor& get(DescriptorDeclHandle h) const;

   private:
    friend class FrameGraph;
    explicit ExecuteContext(const FrameGraph& fg, uint64_t frame_number)
        : m_fg(fg), m_frame_number(frame_number) {
    }
    const FrameGraph& m_fg;
    uint64_t m_frame_number = 0;
};

enum class PassType { Render, Compute };

using ExecuteRenderFn = std::function<void(ExecuteContext&, WGPURenderPassEncoder)>;
using ExecuteComputeFn = std::function<void(ExecuteContext&, WGPUComputePassEncoder)>;

/// Tag type to mark a descriptor slot as dynamic (not auto-set).
struct Dynamic {};
inline constexpr Dynamic dynamic_descriptor{};

class PassBuilder {
   public:
    PassBuilder& color(TextureDeclHandle h);
    PassBuilder& color(TextureDeclHandle h, uint32_t layer);
    PassBuilder& color(WGPUTextureView view, WGPUColor clear_color = {});
    PassBuilder& depth(TextureDeclHandle h);
    PassBuilder& depth(TextureDeclHandle h, uint32_t layer);
    PassBuilder& depth(WGPUTextureView view, float clear_value = 1.0f);
    PassBuilder& depth_readonly(TextureDeclHandle h);
    PassBuilder& present();
    PassBuilder& read(TextureDeclHandle h);
    PassBuilder& storage_write(TextureDeclHandle h);

    /// Declare a descriptor for this pass at the given group index.
    /// Static descriptors are auto-set before the execute callback.
    PassBuilder& descriptor(uint32_t index, DescriptorDeclHandle h);
    /// Declare a dynamic descriptor -- resolved but NOT auto-set. The execute
    /// lambda must call setBindGroup manually (e.g. for per-draw offsets).
    PassBuilder& descriptor(uint32_t index, DescriptorDeclHandle h, Dynamic);

    void execute(ExecuteRenderFn fn);
    void execute(ExecuteComputeFn fn);

   private:
    friend class FrameGraph;
    explicit PassBuilder(FrameGraph& graph, uint32_t pass_index);

    FrameGraph& m_graph;
    uint32_t m_pass_index;
};

class DescriptorBuilder {
   public:
    DescriptorBuilder& buffer(uint32_t binding, BufferDeclHandle h, uint64_t offset = 0,
                              uint64_t size = 0);
    DescriptorBuilder& texture(uint32_t binding, TextureDeclHandle h, uint32_t layer = UINT32_MAX);
    DescriptorBuilder& external_view(uint32_t binding, WGPUTextureView view);
    DescriptorBuilder& external_buffer(uint32_t binding, WGPUBuffer buf, uint64_t offset = 0,
                                       uint64_t size = 0);
    DescriptorBuilder& sampler(uint32_t binding, WGPUSampler sampler);
    [[nodiscard]] DescriptorDeclHandle build();

   private:
    friend class FrameGraph;
    DescriptorBuilder(FrameGraph& fg, std::string name, WGPUBindGroupLayout layout);

    FrameGraph& m_fg;
    std::string m_name;
    WGPUBindGroupLayout m_layout = nullptr;
    std::vector<DescriptorEntry> m_entries;
};

/// Vertex buffer layout description for pipeline cache builders.
struct VertexBufferInfo {
    uint64_t stride = 0;
    WGPUVertexStepMode step_mode = WGPUVertexStepMode_Vertex;
    std::vector<WGPUVertexAttribute> attributes;
};

/// Fluent builder for cached render pipelines.
/// Returned by FrameGraph::render_pipeline(). Call build() to get/create the pipeline.
class RenderPipelineCacheBuilder {
   public:
    auto shader(std::string_view resource_key) -> RenderPipelineCacheBuilder&;
    auto shader_module(WGPUShaderModule module) -> RenderPipelineCacheBuilder&;
    auto vertex_entry(std::string_view name) -> RenderPipelineCacheBuilder&;
    auto fragment_entry(std::string_view name) -> RenderPipelineCacheBuilder&;
    auto color_format(WGPUTextureFormat format, uint32_t index = 0) -> RenderPipelineCacheBuilder&;
    auto topology(WGPUPrimitiveTopology topo) -> RenderPipelineCacheBuilder&;
    auto cull_mode(WGPUCullMode mode) -> RenderPipelineCacheBuilder&;
    auto front_face(WGPUFrontFace face) -> RenderPipelineCacheBuilder&;
    auto blend_state(const WGPUBlendState& blend, uint32_t index = 0)
        -> RenderPipelineCacheBuilder&;
    auto write_mask(WGPUColorWriteMask mask, uint32_t index = 0) -> RenderPipelineCacheBuilder&;
    auto depth_format(WGPUTextureFormat format) -> RenderPipelineCacheBuilder&;
    auto depth_write(bool enabled) -> RenderPipelineCacheBuilder&;
    auto depth_compare(WGPUCompareFunction func) -> RenderPipelineCacheBuilder&;
    auto depth_bias(int32_t constant, float slope_scale) -> RenderPipelineCacheBuilder&;
    auto sample_count(uint32_t count) -> RenderPipelineCacheBuilder&;
    auto vertex_buffer(VertexBufferInfo info) -> RenderPipelineCacheBuilder&;
    auto pipeline_layout(WGPUPipelineLayout layout) -> RenderPipelineCacheBuilder&;
    auto bind_group_layouts(std::initializer_list<WGPUBindGroupLayout> layouts)
        -> RenderPipelineCacheBuilder&;
    auto no_fragment() -> RenderPipelineCacheBuilder&;

    template <typename VertexLayoutT>
    auto vertex_layout() -> RenderPipelineCacheBuilder& {
        VertexBufferInfo info;
        info.stride = VertexLayoutT::stride;
        info.step_mode = VertexLayoutT::step_mode;
        info.attributes.reserve(VertexLayoutT::attributes.size());
        for (const auto& attr : VertexLayoutT::attributes) {
            info.attributes.push_back(attr);
        }
        return vertex_buffer(std::move(info));
    }

    [[nodiscard]] auto build() -> WGPURenderPipeline;

   private:
    friend class FrameGraph;
    RenderPipelineCacheBuilder(FrameGraph& fg, std::string name);

    void ensure_target_count(uint32_t index);

    FrameGraph& m_fg;
    std::string m_name;
    WGPUShaderModule m_shader_module = nullptr;
    // Version of the shader module (from m_fg.m_shader_cache) this pipeline
    // was built against. Snapshot at shader()/shader_module() time.
    uint64_t m_shader_module_version = 0;
    // Name the shader was resolved from (empty for shader_module()). Used
    // only for diagnostics.
    std::string m_shader_resource_key;
    std::string m_vertex_entry = "vs_main";
    std::string m_fragment_entry = "fs_main";

    struct ColorTargetInfo {
        WGPUTextureFormat format = WGPUTextureFormat_BGRA8Unorm;
        WGPUColorWriteMask write_mask = WGPUColorWriteMask_All;
        bool has_blend = false;
        WGPUBlendState blend = {};
    };
    std::vector<ColorTargetInfo> m_color_targets;

    WGPUPrimitiveTopology m_topology = WGPUPrimitiveTopology_TriangleList;
    WGPUCullMode m_cull_mode = WGPUCullMode_None;
    WGPUFrontFace m_front_face = WGPUFrontFace_CCW;
    WGPUTextureFormat m_depth_format = WGPUTextureFormat_Undefined;
    bool m_depth_write = false;
    WGPUCompareFunction m_depth_compare = WGPUCompareFunction_Always;
    int32_t m_depth_bias = 0;
    float m_depth_bias_slope_scale = 0.0f;
    uint32_t m_sample_count = 1;
    std::vector<VertexBufferInfo> m_vertex_buffers;
    WGPUPipelineLayout m_pipeline_layout = nullptr;
    std::vector<WGPUBindGroupLayout> m_bind_group_layouts;
    bool m_has_fragment = true;
};

/// Fluent builder for cached compute pipelines.
class ComputePipelineCacheBuilder {
   public:
    auto shader(std::string_view resource_key) -> ComputePipelineCacheBuilder&;
    auto shader_module(WGPUShaderModule module) -> ComputePipelineCacheBuilder&;
    auto entry_point(std::string_view name) -> ComputePipelineCacheBuilder&;
    auto pipeline_layout(WGPUPipelineLayout layout) -> ComputePipelineCacheBuilder&;
    auto bind_group_layouts(std::initializer_list<WGPUBindGroupLayout> layouts)
        -> ComputePipelineCacheBuilder&;

    [[nodiscard]] auto build() -> WGPUComputePipeline;

   private:
    friend class FrameGraph;
    ComputePipelineCacheBuilder(FrameGraph& fg, std::string name);

    FrameGraph& m_fg;
    std::string m_name;
    WGPUShaderModule m_shader_module = nullptr;
    uint64_t m_shader_module_version = 0;
    std::string m_shader_resource_key;
    std::string m_entry_point = "cs_main";
    WGPUPipelineLayout m_pipeline_layout = nullptr;
    std::vector<WGPUBindGroupLayout> m_bind_group_layouts;
};

class FrameGraph {
   public:
    explicit FrameGraph(const webgpu::Device& device, std::shared_ptr<spdlog::logger> logger,
                        IShaderCompiler* compiler = nullptr);
    ~FrameGraph();
    NO_COPY_MOVE(FrameGraph);

    // -- Textures --------------------------------------------------------
    /// Register a texture decl. First call allocates a slot; subsequent calls
    /// with the same label update the desc and return the existing handle.
    TextureDeclHandle texture(std::string_view debug_label, TextureDesc desc,
                              Lifetime lifetime = Lifetime::Frame);

    /// Persistent texture with initial upload.
    TextureDeclHandle texture(std::string_view debug_label, const WGPUTextureDescriptor& tex_desc,
                              const void* data, uint64_t data_size, uint32_t bytes_per_row,
                              WGPUTextureViewDimension view_dim = WGPUTextureViewDimension_2D);

    /// Update the desc on an existing texture decl (e.g. viewport resize).
    /// Preserves accumulated usage flags. Reactivates the slot if evicted.
    void resize(TextureDeclHandle h, TextureDesc new_desc);

    /// Cold-path name lookup (editor/debug use only). Returns invalid handle
    /// if no decl with that label exists.
    [[nodiscard]] TextureDeclHandle find_texture(std::string_view label) const;

    /// Check whether a handle still refers to an active decl.
    [[nodiscard]] bool valid(TextureDeclHandle h) const;

    /// Access the compiled texture outside of execute lambdas. Returns nullptr
    /// if the decl is not compiled (e.g. not yet materialized this frame).
    [[nodiscard]] const Texture* compiled_texture(TextureDeclHandle h) const;
    [[nodiscard]] const Buffer* compiled_buffer(BufferDeclHandle h) const;
    [[nodiscard]] const Descriptor* compiled_descriptor(DescriptorDeclHandle h) const;

    // -- Buffers ---------------------------------------------------------
    BufferDeclHandle buffer(std::string_view debug_label, BufferDesc desc,
                            Lifetime lifetime = Lifetime::Frame);
    /// Persistent buffer with initial upload.
    BufferDeclHandle buffer(std::string_view debug_label, BufferDesc desc, const void* data);
    /// Wrap an externally-owned buffer. Persistent lifetime.
    BufferDeclHandle import_buffer(std::string_view debug_label, WGPUBuffer buf, std::size_t size,
                                   uint64_t external_version);
    /// Handle-based update for an imported buffer (avoids string lookup).
    void import_buffer(BufferDeclHandle h, WGPUBuffer buf, std::size_t size,
                       uint64_t external_version);

    void resize(BufferDeclHandle h, BufferDesc new_desc);

    [[nodiscard]] BufferDeclHandle find_buffer(std::string_view label) const;
    [[nodiscard]] bool valid(BufferDeclHandle h) const;

    // -- Descriptors -----------------------------------------------------
    DescriptorBuilder descriptor(std::string_view name, WGPUBindGroupLayout layout);
    DescriptorBuilder descriptor(const IPass* pass, WGPUBindGroupLayout layout,
                                 const char* label = nullptr);

    [[nodiscard]] DescriptorDeclHandle find_descriptor(std::string_view name) const;
    [[nodiscard]] bool valid(DescriptorDeclHandle h) const;

    // -- Pass-based API (auto-namespaces by pass name) -------------------
    TextureDeclHandle texture(const IPass* pass, TextureDesc desc, const char* label = nullptr);
    BufferDeclHandle buffer(const IPass* pass, BufferDesc desc, const char* label = nullptr);
    BufferDeclHandle import_buffer(const IPass* pass, WGPUBuffer buf, std::size_t size,
                                   uint64_t external_version, const char* label = nullptr);

    PassBuilder add_pass(std::string name);

    // -- Frame lifecycle -------------------------------------------------
    void begin_frame();
    void compile();
    void execute(WGPUCommandEncoder encoder);

    [[nodiscard]] FallbackPool& fallback_pool();

    [[nodiscard]] const webgpu::Device& device() const {
        return m_device;
    }

    [[nodiscard]] uint64_t frame_number() const {
        return m_frame_number;
    }

    // -- Samplers / BGLs / Shaders / Pipelines ---------------------------
    WGPUSampler sampler(WGPUSamplerBindingType type,
                        WGPUAddressMode address = WGPUAddressMode_ClampToEdge,
                        WGPUMipmapFilterMode mipmap = WGPUMipmapFilterMode_Nearest);

    /// Register a caller-constructed bind group layout under `name` so
    /// downstream FG machinery (pipeline cache, dep tracking) can reference
    /// it by name. The caller retains nothing -- ownership transfers to the
    /// FG cache, which destroys the layout when the cache entry is evicted
    /// or the FG is torn down. Intended for layouts produced by shader
    /// reflection (via the generated `<shader>::create_bind_group_layout_N`
    /// helpers). If `name` is already cached, the supplied `existing` is
    /// released and the cached handle is returned -- callers that register
    /// the same name later are expected to pass a structurally equivalent
    /// layout.
    WGPUBindGroupLayout bind_group_layout(std::string_view name, WGPUBindGroupLayout existing);

    /// Look up a bind group layout that was previously registered via the
    /// (name, existing) overload. Fails loud if `name` is not present --
    /// callers that need the layout must ensure the owning pass registered
    /// it first.
    WGPUBindGroupLayout bind_group_layout(std::string_view name);

    WGPUShaderModule shader(std::string_view resource_key);
    WGPUShaderModule shader_from_wgsl(std::string_view cache_key, const std::string& wgsl_source);
    /// Get-or-build a preprocessor variant of a registered shader. Uses the
    /// base source's revision as the dep, so repeated calls within a session
    /// hit the cache (critical for per-frame callers like load_pass_shader_module
    /// in hot-reload builds -- without this, Slang would recompile every frame).
    WGPUShaderModule shader_variant(std::string_view variant_cache_key,
                                    std::string_view source_resource_key,
                                    boost::span<const std::string_view> defines);
    void invalidate_shader(std::string_view resource_key);
    void invalidate_all_shaders();

    RenderPipelineCacheBuilder render_pipeline(std::string_view name);
    ComputePipelineCacheBuilder compute_pipeline(std::string_view name);
    [[nodiscard]] WGPURenderPipeline get_render_pipeline(std::string_view name) const;
    [[nodiscard]] WGPUComputePipeline get_compute_pipeline(std::string_view name) const;

    // -- Introspection ---------------------------------------------------
    [[nodiscard]] size_t cached_texture_count() const;
    [[nodiscard]] size_t cached_buffer_count() const;
    [[nodiscard]] size_t cached_descriptor_count() const;
    [[nodiscard]] size_t cached_shader_count() const {
        return m_shader_cache.size();
    }
    [[nodiscard]] size_t cached_pipeline_count() const {
        return m_render_pipeline_cache.size() + m_compute_pipeline_cache.size();
    }
    [[nodiscard]] size_t cached_bind_group_layout_count() const {
        return m_bgl_cache.size();
    }

    // Version accessors for use as dep sources by caches external to FG
    // (and by the pipeline caches internally).
    [[nodiscard]] uint64_t shader_version(std::string_view resource_key) const {
        return m_shader_cache.version(resource_key);
    }
    [[nodiscard]] uint64_t bgl_version(WGPUBindGroupLayout layout) const;

   private:
    friend class PassBuilder;
    friend class DescriptorBuilder;
    friend class RenderPipelineCacheBuilder;
    friend class ComputePipelineCacheBuilder;
    friend class ExecuteContext;

    struct ColorAttachmentInfo {
        TextureDeclHandle handle;
        WGPUTextureView external_view = nullptr;
        WGPUColor external_clear{};
        uint32_t layer = UINT32_MAX;
        bool is_read = false;
        bool is_write = false;
        WGPULoadOp load_op = WGPULoadOp_Clear;
        WGPUStoreOp store_op = WGPUStoreOp_Store;
    };

    struct DepthAttachmentInfo {
        TextureDeclHandle handle;
        WGPUTextureView external_view = nullptr;
        float external_clear_value = 1.0f;
        uint32_t layer = UINT32_MAX;
        bool is_read = false;
        bool is_write = false;
    };

    struct DescriptorSlot {
        uint32_t index = 0;
        DescriptorDeclHandle handle;
        bool is_dynamic = false;
    };

    struct Pass {
        std::string name;
        uint32_t index = 0;
        PassType type = PassType::Render;
        std::vector<ColorAttachmentInfo> color_attachments;
        DepthAttachmentInfo depth_attachment;
        bool has_depth = false;
        bool is_present = false;
        std::vector<TextureDeclHandle> reads;
        std::vector<DescriptorSlot> descriptor_slots;
        ExecuteRenderFn render_fn;
        ExecuteComputeFn compute_fn;

        WGPULoadOp depth_load_op = WGPULoadOp_Clear;
        WGPUStoreOp depth_store_op = WGPUStoreOp_Store;
        bool depth_read_only = false;
    };

    /// Scan passes to mark liveness on all referenced decl handles.
    void mark_liveness();
    void materialize_textures();
    void materialize_buffers();
    void materialize_descriptors();
    void evict_unused();

    [[nodiscard]] TextureDecl& tex_decl(TextureDeclHandle h);
    [[nodiscard]] const TextureDecl& tex_decl(TextureDeclHandle h) const;
    [[nodiscard]] BufferDecl& buf_decl(BufferDeclHandle h);
    [[nodiscard]] const BufferDecl& buf_decl(BufferDeclHandle h) const;
    [[nodiscard]] DescriptorDecl& desc_decl(DescriptorDeclHandle h);
    [[nodiscard]] const DescriptorDecl& desc_decl(DescriptorDeclHandle h) const;

    [[nodiscard]] WGPUTextureView resolve_view(const ColorAttachmentInfo& att) const;
    [[nodiscard]] WGPUTextureView resolve_view(const DepthAttachmentInfo& att) const;

    enum class ResourceKind { Texture, Buffer, Descriptor };
    std::string make_pass_key(const IPass* pass, const char* label, ResourceKind kind);

    uint64_t next_version() {
        return m_next_version++;
    }

    const webgpu::Device& m_device;
    IShaderCompiler* m_compiler = nullptr;
    std::shared_ptr<spdlog::logger> m_logger;
    std::unique_ptr<FallbackPool> m_fallback_pool;

    uint64_t m_frame_number = 0;
    uint64_t m_next_version = 1;

    // Decls -- dense vectors indexed by handle.value
    std::vector<TextureDecl> m_texture_decls;
    std::vector<BufferDecl> m_buffer_decls;
    std::vector<DescriptorDecl> m_descriptor_decls;

    // Name -> handle registries. Flat-map + transparent hash -> string_view
    // lookups do not allocate a std::string on the hot path.
    FlatStringMap<uint32_t> m_texture_name_to_handle;
    FlatStringMap<uint32_t> m_buffer_name_to_handle;
    FlatStringMap<uint32_t> m_descriptor_name_to_handle;

    // Compiled resources -- parallel vectors indexed by handle.value
    std::vector<std::unique_ptr<Texture>> m_compiled_textures;
    std::vector<std::unique_ptr<Buffer>> m_compiled_buffers;
    // Descriptors live in m_descriptor_cache (DepTrackedSlotMap, keyed by
    // handle.value) so dep-based invalidation and version tracking are
    // uniform across FG caches.

    // Deferred destruction -- old compiled resources kept alive through execute()
    // so pre-compile references (e.g. ImGui draw data) stay valid. Cleared at
    // begin_frame() after the previous frame's GPU work is submitted.
    std::vector<std::unique_ptr<Texture>> m_deferred_textures;
    std::vector<std::unique_ptr<Buffer>> m_deferred_buffers;

    std::vector<Pass> m_passes;

    using ShaderCache =
        pts::container::DepTrackedSlotMap<std::string, WGPUShaderModule, std::less<>>;
    using BglCache =
        pts::container::DepTrackedSlotMap<std::string, WGPUBindGroupLayout, std::less<>>;
    using RenderPipelineCache =
        pts::container::DepTrackedSlotMap<std::string, WGPURenderPipeline, std::less<>>;
    using ComputePipelineCache =
        pts::container::DepTrackedSlotMap<std::string, WGPUComputePipeline, std::less<>>;
    using DescriptorCache =
        pts::container::DepTrackedSlotMap<uint32_t, std::unique_ptr<Descriptor>>;

    ShaderCache m_shader_cache;
    BglCache m_bgl_cache;
    RenderPipelineCache m_render_pipeline_cache;
    ComputePipelineCache m_compute_pipeline_cache;
    DescriptorCache m_descriptor_cache;

    // Inverse lookup: WGPUBindGroupLayout -> version from m_bgl_cache. Maintained
    // alongside BGL inserts so pipeline builders (which hold raw layout handles
    // rather than names) can gather BGL versions for their dep vector.
    std::unordered_map<WGPUBindGroupLayout, uint64_t> m_bgl_version_lookup;

    using SamplerKey = std::tuple<WGPUSamplerBindingType, WGPUAddressMode, WGPUMipmapFilterMode>;
    std::map<SamplerKey, WGPUSampler> m_sampler_cache;

    // Per-pass auto-naming counters, reset each begin_frame()
    struct PassCounters {
        uint32_t texture = 0;
        uint32_t buffer = 0;
        uint32_t descriptor = 0;
    };
    std::unordered_map<std::string_view, PassCounters> m_pass_counters;
};

}  // namespace pts::rendering
