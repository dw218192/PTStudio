#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/fallbackPool.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/shaderCompiler.h>
#include <core/rendering/webgpu/device.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <spdlog/spdlog.h>

#include <functional>
#include <stdexcept>

namespace pts::rendering {

// ── Compiled resource destructors ────────────────────────────────────────

Texture::~Texture() {
    for (auto lv : layer_views) {
        wgpuTextureViewRelease(lv);
    }
    if (view) {
        wgpuTextureViewRelease(view);
    }
    if (texture) {
        wgpuTextureRelease(texture);
    }
}

Buffer::~Buffer() {
    if (owned && buffer) {
        wgpuBufferDestroy(buffer);
        wgpuBufferRelease(buffer);
    }
}

Descriptor::~Descriptor() {
    if (bind_group) {
        wgpuBindGroupRelease(bind_group);
    }
}

// ── Handle accessors ────────────────────────────────────────────────────

TextureDecl& FrameGraph::tex_decl(TextureDeclHandle h) {
    PRECONDITION(h && h.value < m_texture_decls.size());
    return m_texture_decls[h.value];
}

const TextureDecl& FrameGraph::tex_decl(TextureDeclHandle h) const {
    PRECONDITION(h && h.value < m_texture_decls.size());
    return m_texture_decls[h.value];
}

BufferDecl& FrameGraph::buf_decl(BufferDeclHandle h) {
    PRECONDITION(h && h.value < m_buffer_decls.size());
    return m_buffer_decls[h.value];
}

const BufferDecl& FrameGraph::buf_decl(BufferDeclHandle h) const {
    PRECONDITION(h && h.value < m_buffer_decls.size());
    return m_buffer_decls[h.value];
}

DescriptorDecl& FrameGraph::desc_decl(DescriptorDeclHandle h) {
    PRECONDITION(h && h.value < m_descriptor_decls.size());
    return m_descriptor_decls[h.value];
}

const DescriptorDecl& FrameGraph::desc_decl(DescriptorDeclHandle h) const {
    PRECONDITION(h && h.value < m_descriptor_decls.size());
    return m_descriptor_decls[h.value];
}

// ── ExecuteContext ───────────────────────────────────────────────────────

const Texture& ExecuteContext::get(TextureDeclHandle h) const {
    PRECONDITION_MSG(h, "ExecuteContext::get(TextureDeclHandle): invalid handle");
    PRECONDITION_MSG(h.value < m_fg.m_texture_decls.size(),
                     "ExecuteContext::get(TextureDeclHandle): handle out of range");
    auto& decl = m_fg.m_texture_decls[h.value];
    PRECONDITION_MSG(decl.active, "ExecuteContext::get(TextureDeclHandle): decl not active");
    PRECONDITION_MSG(decl.last_active_frame == m_frame_number,
                     "ExecuteContext::get(TextureDeclHandle): stale handle — not "
                     "referenced by any pass this frame");
    PRECONDITION_MSG(decl.compiled != nullptr,
                     "ExecuteContext::get(TextureDeclHandle): decl has no compiled resource");
    return *decl.compiled;
}

const Buffer& ExecuteContext::get(BufferDeclHandle h) const {
    PRECONDITION_MSG(h, "ExecuteContext::get(BufferDeclHandle): invalid handle");
    PRECONDITION_MSG(h.value < m_fg.m_buffer_decls.size(),
                     "ExecuteContext::get(BufferDeclHandle): handle out of range");
    auto& decl = m_fg.m_buffer_decls[h.value];
    PRECONDITION_MSG(decl.active, "ExecuteContext::get(BufferDeclHandle): decl not active");
    PRECONDITION_MSG(decl.last_active_frame == m_frame_number,
                     "ExecuteContext::get(BufferDeclHandle): stale handle");
    PRECONDITION_MSG(decl.compiled != nullptr,
                     "ExecuteContext::get(BufferDeclHandle): decl has no compiled resource");
    return *decl.compiled;
}

const Descriptor& ExecuteContext::get(DescriptorDeclHandle h) const {
    PRECONDITION_MSG(h, "ExecuteContext::get(DescriptorDeclHandle): invalid handle");
    PRECONDITION_MSG(h.value < m_fg.m_descriptor_decls.size(),
                     "ExecuteContext::get(DescriptorDeclHandle): handle out of range");
    auto& decl = m_fg.m_descriptor_decls[h.value];
    PRECONDITION_MSG(decl.active, "ExecuteContext::get(DescriptorDeclHandle): decl not active");
    PRECONDITION_MSG(decl.last_active_frame == m_frame_number,
                     "ExecuteContext::get(DescriptorDeclHandle): stale handle");
    PRECONDITION_MSG(decl.compiled != nullptr,
                     "ExecuteContext::get(DescriptorDeclHandle): decl has no compiled resource");
    return *decl.compiled;
}

// ── DescriptorBuilder ────────────────────────────────────────────────────

DescriptorBuilder::DescriptorBuilder(FrameGraph& fg, std::string name, WGPUBindGroupLayout layout)
    : m_fg(fg), m_name(std::move(name)), m_layout(layout) {
}

DescriptorBuilder& DescriptorBuilder::buffer(uint32_t binding, BufferDeclHandle h, uint64_t offset,
                                             uint64_t size) {
    PRECONDITION_MSG(h, "DescriptorBuilder::buffer: invalid handle");
    m_entries.push_back({binding, ManagedBufferBinding{h, offset, size}});
    return *this;
}

DescriptorBuilder& DescriptorBuilder::texture(uint32_t binding, TextureDeclHandle h,
                                              uint32_t layer) {
    PRECONDITION_MSG(h, "DescriptorBuilder::texture: invalid handle");
    // Binding a texture in a descriptor implies it will be sampled.
    auto& decl = m_fg.tex_decl(h);
    decl.desc.usage =
        static_cast<WGPUTextureUsage>(decl.desc.usage | WGPUTextureUsage_TextureBinding);
    m_entries.push_back({binding, ManagedTextureBinding{h, layer}});
    return *this;
}

DescriptorBuilder& DescriptorBuilder::external_view(uint32_t binding, WGPUTextureView view) {
    m_entries.push_back({binding, ExternalViewBinding{view}});
    return *this;
}

DescriptorBuilder& DescriptorBuilder::external_buffer(uint32_t binding, WGPUBuffer buf,
                                                      uint64_t offset, uint64_t size) {
    m_entries.push_back({binding, ExternalBufferBinding{buf, offset, size}});
    return *this;
}

DescriptorBuilder& DescriptorBuilder::sampler(uint32_t binding, WGPUSampler sampler) {
    m_entries.push_back({binding, SamplerBinding{sampler}});
    return *this;
}

DescriptorDeclHandle DescriptorBuilder::build() {
    PTS_ZONE_SCOPED;
    PRECONDITION_MSG(m_layout != nullptr, "DescriptorBuilder::build: layout must not be null");

    auto it = m_fg.m_descriptor_name_to_handle.find(std::string_view{m_name});
    uint32_t idx;
    if (it != m_fg.m_descriptor_name_to_handle.end()) {
        idx = it->second;
    } else {
        idx = static_cast<uint32_t>(m_fg.m_descriptor_decls.size());
        m_fg.m_descriptor_decls.emplace_back();
        m_fg.m_descriptor_decls[idx].debug_label = m_name;
        m_fg.m_descriptor_name_to_handle.emplace(m_name, idx);
    }
    auto& decl = m_fg.m_descriptor_decls[idx];
    decl.active = true;
    decl.last_active_frame = m_fg.m_frame_number;
    decl.layout = m_layout;
    decl.entries = std::move(m_entries);
    // Transitively keep referenced resources alive: calling build() counts as
    // usage of every bound managed buffer/texture, so passes that only consume
    // via ExecuteContext::get(descriptor_handle) still keep the inputs live.
    for (auto& entry : decl.entries) {
        std::visit(
            [&](auto& b) {
                using T = std::decay_t<decltype(b)>;
                if constexpr (std::is_same_v<T, ManagedBufferBinding>) {
                    if (b.handle) {
                        m_fg.m_buffer_decls[b.handle.value].last_active_frame = m_fg.m_frame_number;
                    }
                } else if constexpr (std::is_same_v<T, ManagedTextureBinding>) {
                    if (b.handle) {
                        m_fg.m_texture_decls[b.handle.value].last_active_frame =
                            m_fg.m_frame_number;
                    }
                }
            },
            entry.resource);
    }
    return DescriptorDeclHandle{idx};
}

// ── PassBuilder ──────────────────────────────────────────────────────────

PassBuilder::PassBuilder(FrameGraph& graph, uint32_t pass_index)
    : m_graph(graph), m_pass_index(pass_index) {
}

PassBuilder& PassBuilder::color(TextureDeclHandle h) {
    PRECONDITION_MSG(h, "PassBuilder::color: invalid handle");
    auto& decl = m_graph.tex_decl(h);
    auto& pass = m_graph.m_passes[m_pass_index];
    if (decl.first_writer == UINT32_MAX) {
        decl.first_writer = m_pass_index;
    }
    decl.desc.usage =
        static_cast<WGPUTextureUsage>(decl.desc.usage | WGPUTextureUsage_RenderAttachment);
    pass.color_attachments.push_back({h, nullptr, {}, UINT32_MAX, false, true});
    return *this;
}

PassBuilder& PassBuilder::color(TextureDeclHandle h, uint32_t layer) {
    PRECONDITION_MSG(h, "PassBuilder::color: invalid handle");
    auto& decl = m_graph.tex_decl(h);
    PRECONDITION_MSG(decl.desc.array_layers > 1 || decl.desc.force_array_view,
                     "color(h, layer) requires an array texture");
    PRECONDITION_MSG(layer < decl.desc.array_layers, "layer index out of range");
    auto& pass = m_graph.m_passes[m_pass_index];
    if (decl.first_writer == UINT32_MAX) {
        decl.first_writer = m_pass_index;
    }
    decl.desc.usage =
        static_cast<WGPUTextureUsage>(decl.desc.usage | WGPUTextureUsage_RenderAttachment);
    pass.color_attachments.push_back({h, nullptr, {}, layer, false, true});
    return *this;
}

PassBuilder& PassBuilder::color(WGPUTextureView view, WGPUColor clear_color) {
    auto& pass = m_graph.m_passes[m_pass_index];
    FrameGraph::ColorAttachmentInfo info;
    info.external_view = view;
    info.external_clear = clear_color;
    info.is_write = true;
    pass.color_attachments.push_back(info);
    return *this;
}

PassBuilder& PassBuilder::depth(TextureDeclHandle h) {
    PRECONDITION_MSG(h, "PassBuilder::depth: invalid handle");
    auto& decl = m_graph.tex_decl(h);
    auto& pass = m_graph.m_passes[m_pass_index];
    if (decl.first_writer == UINT32_MAX) {
        decl.first_writer = m_pass_index;
    }
    decl.desc.usage =
        static_cast<WGPUTextureUsage>(decl.desc.usage | WGPUTextureUsage_RenderAttachment);
    pass.depth_attachment = {h, nullptr, 1.0f, UINT32_MAX, true, true};
    pass.has_depth = true;
    return *this;
}

PassBuilder& PassBuilder::depth(TextureDeclHandle h, uint32_t layer) {
    PRECONDITION_MSG(h, "PassBuilder::depth: invalid handle");
    auto& decl = m_graph.tex_decl(h);
    PRECONDITION_MSG(decl.desc.array_layers > 1 || decl.desc.force_array_view,
                     "depth(h, layer) requires an array texture");
    PRECONDITION_MSG(layer < decl.desc.array_layers, "layer index out of range");
    auto& pass = m_graph.m_passes[m_pass_index];
    if (decl.first_writer == UINT32_MAX) {
        decl.first_writer = m_pass_index;
    }
    decl.desc.usage =
        static_cast<WGPUTextureUsage>(decl.desc.usage | WGPUTextureUsage_RenderAttachment);
    pass.depth_attachment = {h, nullptr, 1.0f, layer, true, true};
    pass.has_depth = true;
    return *this;
}

PassBuilder& PassBuilder::depth(WGPUTextureView view, float clear_value) {
    auto& pass = m_graph.m_passes[m_pass_index];
    pass.depth_attachment = {};
    pass.depth_attachment.external_view = view;
    pass.depth_attachment.external_clear_value = clear_value;
    pass.depth_attachment.is_write = true;
    pass.has_depth = true;
    return *this;
}

PassBuilder& PassBuilder::depth_readonly(TextureDeclHandle h) {
    PRECONDITION_MSG(h, "PassBuilder::depth_readonly: invalid handle");
    auto& pass = m_graph.m_passes[m_pass_index];
    pass.depth_attachment = {h, nullptr, 1.0f, UINT32_MAX, true, false};
    pass.has_depth = true;
    return *this;
}

PassBuilder& PassBuilder::present() {
    m_graph.m_passes[m_pass_index].is_present = true;
    return *this;
}

PassBuilder& PassBuilder::read(TextureDeclHandle h) {
    PRECONDITION_MSG(h, "PassBuilder::read: invalid handle");
    auto& decl = m_graph.tex_decl(h);
    auto& pass = m_graph.m_passes[m_pass_index];
    pass.reads.push_back(h);
    decl.desc.usage =
        static_cast<WGPUTextureUsage>(decl.desc.usage | WGPUTextureUsage_TextureBinding);
    return *this;
}

PassBuilder& PassBuilder::storage_write(TextureDeclHandle h) {
    PRECONDITION_MSG(h, "PassBuilder::storage_write: invalid handle");
    auto& decl = m_graph.tex_decl(h);
    auto& pass = m_graph.m_passes[m_pass_index];
    if (decl.first_writer == UINT32_MAX) {
        decl.first_writer = m_pass_index;
    }
    decl.desc.usage =
        static_cast<WGPUTextureUsage>(decl.desc.usage | WGPUTextureUsage_StorageBinding);
    pass.reads.push_back(h);
    return *this;
}

PassBuilder& PassBuilder::descriptor(uint32_t index, DescriptorDeclHandle h) {
    PRECONDITION_MSG(h, "PassBuilder::descriptor: invalid handle");
    auto& pass = m_graph.m_passes[m_pass_index];
    pass.descriptor_slots.push_back({index, h, false});
    return *this;
}

PassBuilder& PassBuilder::descriptor(uint32_t index, DescriptorDeclHandle h, Dynamic) {
    PRECONDITION_MSG(h, "PassBuilder::descriptor: invalid handle");
    auto& pass = m_graph.m_passes[m_pass_index];
    pass.descriptor_slots.push_back({index, h, true});
    return *this;
}

void PassBuilder::execute(ExecuteRenderFn fn) {
    auto& pass = m_graph.m_passes[m_pass_index];
    pass.type = PassType::Render;
    pass.render_fn = std::move(fn);
}

void PassBuilder::execute(ExecuteComputeFn fn) {
    auto& pass = m_graph.m_passes[m_pass_index];
    pass.type = PassType::Compute;
    pass.compute_fn = std::move(fn);
}

// ── FrameGraph ───────────────────────────────────────────────────────────

FrameGraph::FrameGraph(const webgpu::Device& device, std::shared_ptr<spdlog::logger> logger,
                       IShaderCompiler* compiler)
    : m_device(device), m_compiler(compiler), m_logger(std::move(logger)) {
}

FrameGraph::~FrameGraph() {
    // Release pipelines before shaders (pipelines reference shaders)
    m_render_pipeline_cache.for_each([](const std::string&, WGPURenderPipeline& p) {
        if (p) wgpuRenderPipelineRelease(p);
    });
    m_render_pipeline_cache.clear();
    m_compute_pipeline_cache.for_each([](const std::string&, WGPUComputePipeline& p) {
        if (p) wgpuComputePipelineRelease(p);
    });
    m_compute_pipeline_cache.clear();
    m_shader_cache.for_each([](const std::string&, WGPUShaderModule& m) {
        if (m) wgpuShaderModuleRelease(m);
    });
    m_shader_cache.clear();
    for (auto& [key, s] : m_sampler_cache) {
        wgpuSamplerRelease(s);
    }
    m_sampler_cache.clear();
    m_bgl_cache.for_each([](const std::string&, WGPUBindGroupLayout& bgl) {
        if (bgl) wgpuBindGroupLayoutRelease(bgl);
    });
    m_bgl_cache.clear();
    m_bgl_version_lookup.clear();
    // Destroy compiled resources before decls
    m_descriptor_cache.clear();
    m_compiled_buffers.clear();
    m_compiled_textures.clear();
    m_descriptor_decls.clear();
    m_buffer_decls.clear();
    m_texture_decls.clear();
    m_fallback_pool.reset();
}

WGPUSampler FrameGraph::sampler(WGPUSamplerBindingType type, WGPUAddressMode address,
                                WGPUMipmapFilterMode mipmap) {
    PTS_ZONE_SCOPED;
    auto key = SamplerKey{type, address, mipmap};
    auto it = m_sampler_cache.find(key);
    if (it != m_sampler_cache.end()) return it->second;

    WGPUSamplerDescriptor desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    auto filter =
        (type == WGPUSamplerBindingType_Filtering) ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    desc.magFilter = filter;
    desc.minFilter = filter;
    desc.mipmapFilter = mipmap;
    desc.addressModeU = address;
    desc.addressModeV = address;
    desc.addressModeW = address;

    auto s = wgpuDeviceCreateSampler(m_device.handle(), &desc);
    INVARIANT_MSG(s, "FrameGraph::sampler() failed to create sampler");
    m_sampler_cache.emplace(key, s);
    return s;
}

WGPUBindGroupLayout FrameGraph::bind_group_layout(std::string_view name,
                                                  WGPUBindGroupLayout existing) {
    PTS_ZONE_SCOPED;
    INVARIANT_MSG(existing, "FrameGraph::bind_group_layout: existing layout must be non-null");
    auto& bgl = m_bgl_cache.get_or_build(
        name, pts::cache::DepTrackedCache<std::string, WGPUBindGroupLayout>::Span{},
        [&] { return existing; });
    if (bgl != existing) {
        // Cache hit on same name but with a different handle: drop the new
        // one — callers are expected to use a stable name per layout identity.
        wgpuBindGroupLayoutRelease(existing);
    }
    m_bgl_version_lookup[bgl] = m_bgl_cache.version(name);
    return bgl;
}

WGPUBindGroupLayout FrameGraph::bind_group_layout(std::string_view name) {
    PTS_ZONE_SCOPED;
    auto* cached = m_bgl_cache.find(name);
    INVARIANT_MSG(cached,
                  "FrameGraph::bind_group_layout(name): no layout registered under this name; "
                  "the owning pass must register it first via the (name, existing) overload");
    return *cached;
}

uint64_t FrameGraph::bgl_version(WGPUBindGroupLayout layout) const {
    auto it = m_bgl_version_lookup.find(layout);
    if (it == m_bgl_version_lookup.end()) return 0;
    return it->second;
}

// ── Shaders ──────────────────────────────────────────────────────────────

WGPUShaderModule FrameGraph::shader(std::string_view resource_key) {
    PTS_ZONE_SCOPED;
    PRECONDITION_MSG(m_compiler, "FrameGraph::shader() requires an IShaderCompiler");
    // Dep: source revision tracked by the compiler. Bumped by invalidate_shader().
    uint64_t rev = m_compiler->source_revision(resource_key);
    uint64_t deps[] = {rev};
    return m_shader_cache.get_or_build_with_replace(
        resource_key, ShaderCache::Span{deps, 1},
        [&]() -> WGPUShaderModule {
            auto wgsl = m_compiler->compile(ShaderKey{resource_key, {}});
            WGPUShaderSourceWGSL wgsl_desc = WGPU_SHADER_SOURCE_WGSL_INIT;
            wgsl_desc.code.data = wgsl.data();
            wgsl_desc.code.length = wgsl.size();
            WGPUShaderModuleDescriptor desc = {};
            desc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl_desc);
            auto m = wgpuDeviceCreateShaderModule(m_device.handle(), &desc);
            INVARIANT_MSG(m, "FrameGraph::shader() failed to create shader module");
            return m;
        },
        [](WGPUShaderModule& old) {
            if (old) wgpuShaderModuleRelease(old);
        });
}

WGPUShaderModule FrameGraph::shader_from_wgsl(std::string_view cache_key,
                                              const std::string& wgsl_source) {
    PTS_ZONE_SCOPED;
    // shader_from_wgsl is used for ad-hoc inline WGSL (e.g. test fixtures); the
    // caller does not use the compiler. Dep is the cache_key's revision as
    // reported by the compiler (1 by default, so cache hits from frame to
    // frame). When no compiler is attached, skip revision tracking entirely.
    uint64_t rev = m_compiler ? m_compiler->source_revision(cache_key) : 1;
    uint64_t deps[] = {rev};
    return m_shader_cache.get_or_build_with_replace(
        cache_key, ShaderCache::Span{deps, 1},
        [&]() -> WGPUShaderModule {
            WGPUShaderSourceWGSL wgsl_desc = WGPU_SHADER_SOURCE_WGSL_INIT;
            wgsl_desc.code.data = wgsl_source.data();
            wgsl_desc.code.length = wgsl_source.size();
            WGPUShaderModuleDescriptor desc = {};
            desc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl_desc);
            auto m = wgpuDeviceCreateShaderModule(m_device.handle(), &desc);
            INVARIANT_MSG(m, "FrameGraph::shader() failed to create shader module");
            return m;
        },
        [](WGPUShaderModule& old) {
            if (old) wgpuShaderModuleRelease(old);
        });
}

WGPUShaderModule FrameGraph::shader_variant(std::string_view variant_cache_key,
                                            std::string_view source_resource_key,
                                            boost::span<const std::string_view> defines) {
    PTS_ZONE_SCOPED;
    PRECONDITION_MSG(m_compiler, "FrameGraph::shader_variant() requires an IShaderCompiler");
    // Dep is the source's revision: when the underlying Slang source changes,
    // all variants built from it must rebuild.
    uint64_t rev = m_compiler->source_revision(source_resource_key);
    uint64_t deps[] = {rev};
    return m_shader_cache.get_or_build_with_replace(
        variant_cache_key, ShaderCache::Span{deps, 1},
        [&]() -> WGPUShaderModule {
            auto wgsl = m_compiler->compile(ShaderKey{source_resource_key, defines});
            WGPUShaderSourceWGSL wgsl_desc = WGPU_SHADER_SOURCE_WGSL_INIT;
            wgsl_desc.code.data = wgsl.data();
            wgsl_desc.code.length = wgsl.size();
            WGPUShaderModuleDescriptor desc = {};
            desc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl_desc);
            auto m = wgpuDeviceCreateShaderModule(m_device.handle(), &desc);
            INVARIANT_MSG(m, "FrameGraph::shader_variant() failed to create shader module");
            return m;
        },
        [](WGPUShaderModule& old) {
            if (old) wgpuShaderModuleRelease(old);
        });
}

void FrameGraph::invalidate_shader(std::string_view resource_key) {
    // Release any existing module and drop entry so next shader() call rebuilds
    // with a fresh version. Bump the source revision on the compiler so any
    // variants of this source (which use the same source_revision as their
    // dep) rebuild too.
    if (auto* m = m_shader_cache.find(resource_key)) {
        if (*m) wgpuShaderModuleRelease(*m);
    }
    m_shader_cache.erase(resource_key);
    if (m_compiler) m_compiler->invalidate(resource_key);
}

void FrameGraph::invalidate_all_shaders() {
    if (m_compiler) {
        m_shader_cache.for_each(
            [this](const std::string& key, WGPUShaderModule&) { m_compiler->invalidate(key); });
    }
    m_shader_cache.for_each([](const std::string&, WGPUShaderModule& m) {
        if (m) wgpuShaderModuleRelease(m);
    });
    m_shader_cache.clear();
}

// ── Pipeline cache ───────────────────────────────────────────────────────

RenderPipelineCacheBuilder::RenderPipelineCacheBuilder(FrameGraph& fg, std::string name)
    : m_fg(fg), m_name(std::move(name)) {
    m_color_targets.push_back({});
}

auto RenderPipelineCacheBuilder::shader(std::string_view resource_key)
    -> RenderPipelineCacheBuilder& {
    m_shader_module = m_fg.shader(resource_key);
    m_shader_resource_key = std::string(resource_key);
    m_shader_module_version = m_fg.m_shader_cache.version(resource_key);
    return *this;
}

auto RenderPipelineCacheBuilder::shader_module(WGPUShaderModule module)
    -> RenderPipelineCacheBuilder& {
    m_shader_module = module;
    m_shader_module_version = 0;
    m_fg.m_shader_cache.for_each([&](const std::string& key, WGPUShaderModule& m) {
        if (m == module) {
            m_shader_module_version = m_fg.m_shader_cache.version(key);
            m_shader_resource_key = key;
        }
    });
    if (m_shader_module_version == 0) {
        // Not in cache — fall back to handle address as a stable identifier.
        m_shader_module_version = reinterpret_cast<uintptr_t>(module);
    }
    return *this;
}

auto RenderPipelineCacheBuilder::vertex_entry(std::string_view name)
    -> RenderPipelineCacheBuilder& {
    m_vertex_entry = std::string(name);
    return *this;
}

auto RenderPipelineCacheBuilder::fragment_entry(std::string_view name)
    -> RenderPipelineCacheBuilder& {
    m_fragment_entry = std::string(name);
    return *this;
}

auto RenderPipelineCacheBuilder::color_format(WGPUTextureFormat format, uint32_t index)
    -> RenderPipelineCacheBuilder& {
    ensure_target_count(index);
    m_color_targets[index].format = format;
    return *this;
}

auto RenderPipelineCacheBuilder::topology(WGPUPrimitiveTopology topo)
    -> RenderPipelineCacheBuilder& {
    m_topology = topo;
    return *this;
}

auto RenderPipelineCacheBuilder::cull_mode(WGPUCullMode mode) -> RenderPipelineCacheBuilder& {
    m_cull_mode = mode;
    return *this;
}

auto RenderPipelineCacheBuilder::front_face(WGPUFrontFace face) -> RenderPipelineCacheBuilder& {
    m_front_face = face;
    return *this;
}

auto RenderPipelineCacheBuilder::blend_state(const WGPUBlendState& blend, uint32_t index)
    -> RenderPipelineCacheBuilder& {
    ensure_target_count(index);
    m_color_targets[index].blend = blend;
    m_color_targets[index].has_blend = true;
    return *this;
}

auto RenderPipelineCacheBuilder::write_mask(WGPUColorWriteMask mask, uint32_t index)
    -> RenderPipelineCacheBuilder& {
    ensure_target_count(index);
    m_color_targets[index].write_mask = mask;
    return *this;
}

auto RenderPipelineCacheBuilder::depth_format(WGPUTextureFormat format)
    -> RenderPipelineCacheBuilder& {
    m_depth_format = format;
    return *this;
}

auto RenderPipelineCacheBuilder::depth_write(bool enabled) -> RenderPipelineCacheBuilder& {
    m_depth_write = enabled;
    return *this;
}

auto RenderPipelineCacheBuilder::depth_compare(WGPUCompareFunction func)
    -> RenderPipelineCacheBuilder& {
    m_depth_compare = func;
    return *this;
}

auto RenderPipelineCacheBuilder::depth_bias(int32_t constant, float slope_scale)
    -> RenderPipelineCacheBuilder& {
    m_depth_bias = constant;
    m_depth_bias_slope_scale = slope_scale;
    return *this;
}

auto RenderPipelineCacheBuilder::sample_count(uint32_t count) -> RenderPipelineCacheBuilder& {
    m_sample_count = count;
    return *this;
}

auto RenderPipelineCacheBuilder::vertex_buffer(VertexBufferInfo info)
    -> RenderPipelineCacheBuilder& {
    m_vertex_buffers.push_back(std::move(info));
    return *this;
}

auto RenderPipelineCacheBuilder::bind_group_layouts(
    std::initializer_list<WGPUBindGroupLayout> layouts) -> RenderPipelineCacheBuilder& {
    m_bind_group_layouts.assign(layouts.begin(), layouts.end());
    return *this;
}

auto RenderPipelineCacheBuilder::pipeline_layout(WGPUPipelineLayout layout)
    -> RenderPipelineCacheBuilder& {
    m_pipeline_layout = layout;
    return *this;
}

auto RenderPipelineCacheBuilder::no_fragment() -> RenderPipelineCacheBuilder& {
    m_has_fragment = false;
    m_color_targets.clear();
    return *this;
}

void RenderPipelineCacheBuilder::ensure_target_count(uint32_t index) {
    auto required = static_cast<size_t>(index) + 1;
    while (m_color_targets.size() < required) {
        m_color_targets.push_back({});
    }
}

auto RenderPipelineCacheBuilder::build() -> WGPURenderPipeline {
    PRECONDITION_MSG(m_shader_module != nullptr, "shader not set on render pipeline builder");

    // Deps: shader module version + every bound BGL's version. Config (blend,
    // formats, vertex layout) is considered constant per pipeline name.
    boost::container::small_vector<uint64_t, 8> deps;
    deps.push_back(m_shader_module_version);
    for (auto bgl : m_bind_group_layouts) {
        deps.push_back(m_fg.bgl_version(bgl));
    }

    return m_fg.m_render_pipeline_cache.get_or_build_with_replace(
        m_name, FrameGraph::RenderPipelineCache::Span{deps.data(), deps.size()},
        [&]() -> WGPURenderPipeline {
            PTS_ZONE_NAMED("render_pipeline build");
            webgpu::RenderPipelineBuilder builder(m_fg.m_device);
            builder.shader(m_shader_module);
            builder.vertex_entry(m_vertex_entry);

            if (!m_has_fragment) {
                builder.no_fragment();
            } else {
                builder.fragment_entry(m_fragment_entry);
                for (uint32_t i = 0; i < static_cast<uint32_t>(m_color_targets.size()); ++i) {
                    builder.color_format(m_color_targets[i].format, i);
                    builder.write_mask(m_color_targets[i].write_mask, i);
                    if (m_color_targets[i].has_blend) {
                        builder.blend_state(m_color_targets[i].blend, i);
                    }
                }
            }

            builder.topology(m_topology);
            builder.cull_mode(m_cull_mode);
            builder.front_face(m_front_face);
            builder.depth_format(m_depth_format);
            builder.depth_write(m_depth_write);
            builder.depth_compare(m_depth_compare);
            builder.depth_bias(m_depth_bias, m_depth_bias_slope_scale);
            builder.sample_count(m_sample_count);

            for (const auto& vb : m_vertex_buffers) {
                webgpu::VertexBufferLayout layout;
                layout.stride = vb.stride;
                layout.step_mode = vb.step_mode;
                layout.attributes = vb.attributes;
                builder.vertex_buffer(std::move(layout));
            }

            WGPUPipelineLayout owned_pl = nullptr;
            if (!m_bind_group_layouts.empty()) {
                PRECONDITION_MSG(m_pipeline_layout == nullptr,
                                 "render_pipeline: pipeline_layout() and bind_group_layouts() "
                                 "are mutually exclusive");
                WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
                pl_desc.bindGroupLayoutCount = static_cast<uint32_t>(m_bind_group_layouts.size());
                pl_desc.bindGroupLayouts = m_bind_group_layouts.data();
                owned_pl = wgpuDeviceCreatePipelineLayout(m_fg.m_device.handle(), &pl_desc);
                INVARIANT_MSG(owned_pl, "render_pipeline: failed to create pipeline layout");
                builder.pipeline_layout(owned_pl);
            } else if (m_pipeline_layout) {
                builder.pipeline_layout(m_pipeline_layout);
            }

            auto raii = builder.build();
            auto handle = raii.handle();
            wgpuRenderPipelineAddRef(handle);

            if (owned_pl) {
                wgpuPipelineLayoutRelease(owned_pl);
            }

            return handle;
        },
        [](WGPURenderPipeline& old) {
            if (old) wgpuRenderPipelineRelease(old);
        });
}

// --- ComputePipelineCacheBuilder ---

ComputePipelineCacheBuilder::ComputePipelineCacheBuilder(FrameGraph& fg, std::string name)
    : m_fg(fg), m_name(std::move(name)) {
}

auto ComputePipelineCacheBuilder::shader(std::string_view resource_key)
    -> ComputePipelineCacheBuilder& {
    m_shader_module = m_fg.shader(resource_key);
    m_shader_resource_key = std::string(resource_key);
    m_shader_module_version = m_fg.m_shader_cache.version(resource_key);
    return *this;
}

auto ComputePipelineCacheBuilder::shader_module(WGPUShaderModule module)
    -> ComputePipelineCacheBuilder& {
    m_shader_module = module;
    m_shader_module_version = 0;
    m_fg.m_shader_cache.for_each([&](const std::string& key, WGPUShaderModule& m) {
        if (m == module) {
            m_shader_module_version = m_fg.m_shader_cache.version(key);
            m_shader_resource_key = key;
        }
    });
    if (m_shader_module_version == 0) {
        m_shader_module_version = reinterpret_cast<uintptr_t>(module);
    }
    return *this;
}

auto ComputePipelineCacheBuilder::entry_point(std::string_view name)
    -> ComputePipelineCacheBuilder& {
    m_entry_point = std::string(name);
    return *this;
}

auto ComputePipelineCacheBuilder::pipeline_layout(WGPUPipelineLayout layout)
    -> ComputePipelineCacheBuilder& {
    m_pipeline_layout = layout;
    return *this;
}

auto ComputePipelineCacheBuilder::bind_group_layouts(
    std::initializer_list<WGPUBindGroupLayout> layouts) -> ComputePipelineCacheBuilder& {
    m_bind_group_layouts.assign(layouts.begin(), layouts.end());
    return *this;
}

auto ComputePipelineCacheBuilder::build() -> WGPUComputePipeline {
    PRECONDITION_MSG(m_shader_module != nullptr, "shader not set on compute pipeline builder");

    boost::container::small_vector<uint64_t, 8> deps;
    deps.push_back(m_shader_module_version);
    for (auto bgl : m_bind_group_layouts) {
        deps.push_back(m_fg.bgl_version(bgl));
    }

    return m_fg.m_compute_pipeline_cache.get_or_build_with_replace(
        m_name, FrameGraph::ComputePipelineCache::Span{deps.data(), deps.size()},
        [&]() -> WGPUComputePipeline {
            PTS_ZONE_NAMED("compute_pipeline build");
            webgpu::ComputePipelineBuilder builder(m_fg.m_device);
            builder.shader(m_shader_module);
            builder.entry_point(m_entry_point);

            WGPUPipelineLayout owned_pl = nullptr;
            if (!m_bind_group_layouts.empty()) {
                PRECONDITION_MSG(m_pipeline_layout == nullptr,
                                 "compute_pipeline: pipeline_layout() and bind_group_layouts() "
                                 "are mutually exclusive");
                WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
                pl_desc.bindGroupLayoutCount = static_cast<uint32_t>(m_bind_group_layouts.size());
                pl_desc.bindGroupLayouts = m_bind_group_layouts.data();
                owned_pl = wgpuDeviceCreatePipelineLayout(m_fg.m_device.handle(), &pl_desc);
                INVARIANT_MSG(owned_pl, "compute_pipeline: failed to create pipeline layout");
                builder.pipeline_layout(owned_pl);
            } else if (m_pipeline_layout) {
                builder.pipeline_layout(m_pipeline_layout);
            }

            auto raii = builder.build();
            auto handle = raii.handle();
            wgpuComputePipelineAddRef(handle);
            if (owned_pl) {
                wgpuPipelineLayoutRelease(owned_pl);
            }
            return handle;
        },
        [](WGPUComputePipeline& old) {
            if (old) wgpuComputePipelineRelease(old);
        });
}

RenderPipelineCacheBuilder FrameGraph::render_pipeline(std::string_view name) {
    return RenderPipelineCacheBuilder(*this, std::string(name));
}

ComputePipelineCacheBuilder FrameGraph::compute_pipeline(std::string_view name) {
    return ComputePipelineCacheBuilder(*this, std::string(name));
}

WGPURenderPipeline FrameGraph::get_render_pipeline(std::string_view name) const {
    auto* p = m_render_pipeline_cache.find(name);
    PRECONDITION_MSG(p != nullptr, "get_render_pipeline: pipeline not found in cache");
    return *p;
}

WGPUComputePipeline FrameGraph::get_compute_pipeline(std::string_view name) const {
    auto* p = m_compute_pipeline_cache.find(name);
    PRECONDITION_MSG(p != nullptr, "get_compute_pipeline: pipeline not found in cache");
    return *p;
}

FallbackPool& FrameGraph::fallback_pool() {
    if (!m_fallback_pool) {
        m_fallback_pool = std::make_unique<FallbackPool>(m_device);
    }
    return *m_fallback_pool;
}

// ── Decl creation / lookup ───────────────────────────────────────────────

TextureDeclHandle FrameGraph::texture(std::string_view debug_label, TextureDesc desc,
                                      Lifetime lifetime) {
    PTS_ZONE_SCOPED;
    auto it = m_texture_name_to_handle.find(debug_label);
    if (it != m_texture_name_to_handle.end()) {
        uint32_t idx = it->second;
        auto& decl = m_texture_decls[idx];
        decl.active = true;
        decl.last_active_frame = m_frame_number;
        auto merged_usage = static_cast<WGPUTextureUsage>(decl.desc.usage | desc.usage);
        decl.desc = desc;
        decl.desc.usage = merged_usage;
        return TextureDeclHandle{idx};
    }
    uint32_t idx = static_cast<uint32_t>(m_texture_decls.size());
    m_texture_decls.emplace_back();
    m_compiled_textures.emplace_back();
    auto& decl = m_texture_decls[idx];
    decl.debug_label = std::string(debug_label);
    decl.desc = desc;
    decl.lifetime = lifetime;
    decl.active = true;
    decl.last_active_frame = m_frame_number;
    m_texture_name_to_handle.emplace(std::string(debug_label), idx);
    return TextureDeclHandle{idx};
}

TextureDeclHandle FrameGraph::texture(std::string_view debug_label,
                                      const WGPUTextureDescriptor& tex_desc, const void* data,
                                      uint64_t data_size, uint32_t bytes_per_row,
                                      WGPUTextureViewDimension view_dim) {
    PTS_ZONE_SCOPED;
    PRECONDITION(data != nullptr);
    PRECONDITION(data_size > 0);
    auto it = m_texture_name_to_handle.find(debug_label);
    if (it != m_texture_name_to_handle.end()) {
        auto& decl = m_texture_decls[it->second];
        decl.active = true;
        decl.last_active_frame = m_frame_number;
        return TextureDeclHandle{it->second};
    }
    uint32_t idx = static_cast<uint32_t>(m_texture_decls.size());
    m_texture_decls.emplace_back();
    m_compiled_textures.emplace_back();
    auto& decl = m_texture_decls[idx];
    decl.debug_label = std::string(debug_label);
    decl.desc.width = tex_desc.size.width;
    decl.desc.height = tex_desc.size.height;
    decl.desc.array_layers = tex_desc.size.depthOrArrayLayers;
    decl.desc.format = tex_desc.format;
    decl.desc.usage = tex_desc.usage;
    decl.desc.force_array_view =
        (view_dim == WGPUTextureViewDimension_2DArray || view_dim == WGPUTextureViewDimension_Cube);
    decl.lifetime = Lifetime::Persistent;
    decl.active = true;
    decl.last_active_frame = m_frame_number;
    decl.upload_data = data;
    decl.upload_size = data_size;
    decl.upload_bytes_per_row = bytes_per_row;
    decl.upload_desc = tex_desc;
    decl.upload_view_dim = view_dim;
    decl.has_upload = true;
    m_texture_name_to_handle.emplace(std::string(debug_label), idx);
    return TextureDeclHandle{idx};
}

void FrameGraph::resize(TextureDeclHandle h, TextureDesc new_desc) {
    PTS_ZONE_SCOPED;
    auto& decl = tex_decl(h);
    decl.active = true;
    decl.last_active_frame = m_frame_number;
    auto merged_usage = static_cast<WGPUTextureUsage>(decl.desc.usage | new_desc.usage);
    decl.desc = new_desc;
    decl.desc.usage = merged_usage;
}

TextureDeclHandle FrameGraph::find_texture(std::string_view label) const {
    auto it = m_texture_name_to_handle.find(label);
    if (it == m_texture_name_to_handle.end()) return TextureDeclHandle{};
    if (!m_texture_decls[it->second].active) return TextureDeclHandle{};
    return TextureDeclHandle{it->second};
}

bool FrameGraph::valid(TextureDeclHandle h) const {
    return h && h.value < m_texture_decls.size() && m_texture_decls[h.value].active;
}

const Texture* FrameGraph::compiled_texture(TextureDeclHandle h) const {
    if (!h || h.value >= m_compiled_textures.size()) return nullptr;
    return m_compiled_textures[h.value].get();
}

const Buffer* FrameGraph::compiled_buffer(BufferDeclHandle h) const {
    if (!h || h.value >= m_compiled_buffers.size()) return nullptr;
    return m_compiled_buffers[h.value].get();
}

const Descriptor* FrameGraph::compiled_descriptor(DescriptorDeclHandle h) const {
    if (!h) return nullptr;
    auto* p = m_descriptor_cache.find(h.value);
    return (p && *p) ? p->get() : nullptr;
}

BufferDeclHandle FrameGraph::buffer(std::string_view debug_label, BufferDesc desc,
                                    Lifetime lifetime) {
    PTS_ZONE_SCOPED;
    auto it = m_buffer_name_to_handle.find(debug_label);
    if (it != m_buffer_name_to_handle.end()) {
        uint32_t idx = it->second;
        auto& decl = m_buffer_decls[idx];
        decl.active = true;
        decl.last_active_frame = m_frame_number;
        if (desc.size > decl.desc.size) {
            decl.desc.size = desc.size;
        }
        decl.desc.usage = static_cast<WGPUBufferUsage>(decl.desc.usage | desc.usage);
        return BufferDeclHandle{idx};
    }
    uint32_t idx = static_cast<uint32_t>(m_buffer_decls.size());
    m_buffer_decls.emplace_back();
    m_compiled_buffers.emplace_back();
    auto& decl = m_buffer_decls[idx];
    decl.debug_label = std::string(debug_label);
    decl.desc = desc;
    decl.lifetime = lifetime;
    decl.active = true;
    decl.last_active_frame = m_frame_number;
    m_buffer_name_to_handle.emplace(std::string(debug_label), idx);
    return BufferDeclHandle{idx};
}

BufferDeclHandle FrameGraph::buffer(std::string_view debug_label, BufferDesc desc,
                                    const void* data) {
    PTS_ZONE_SCOPED;
    PRECONDITION(data != nullptr);
    PRECONDITION_MSG((desc.usage & WGPUBufferUsage_CopyDst) != 0,
                     "buffer(name,desc,data) requires WGPUBufferUsage_CopyDst");
    auto it = m_buffer_name_to_handle.find(debug_label);
    if (it != m_buffer_name_to_handle.end()) {
        auto& decl = m_buffer_decls[it->second];
        decl.active = true;
        decl.last_active_frame = m_frame_number;
        return BufferDeclHandle{it->second};
    }
    uint32_t idx = static_cast<uint32_t>(m_buffer_decls.size());
    m_buffer_decls.emplace_back();
    m_compiled_buffers.emplace_back();
    auto& decl = m_buffer_decls[idx];
    decl.debug_label = std::string(debug_label);
    decl.desc = desc;
    decl.lifetime = Lifetime::Persistent;
    decl.active = true;
    decl.last_active_frame = m_frame_number;
    decl.upload_data = data;
    decl.upload_size = desc.size;
    decl.has_upload = true;
    m_buffer_name_to_handle.emplace(std::string(debug_label), idx);
    return BufferDeclHandle{idx};
}

BufferDeclHandle FrameGraph::import_buffer(std::string_view debug_label, WGPUBuffer buf,
                                           std::size_t size, uint64_t external_version) {
    PTS_ZONE_SCOPED;
    PRECONDITION_MSG(buf != nullptr, "import_buffer: buffer must not be null");
    auto it = m_buffer_name_to_handle.find(debug_label);
    if (it != m_buffer_name_to_handle.end()) {
        uint32_t idx = it->second;
        auto& decl = m_buffer_decls[idx];
        decl.active = true;
        decl.last_active_frame = m_frame_number;
        decl.external_buffer = buf;
        decl.external_size = size;
        decl.external_version = external_version;
        return BufferDeclHandle{idx};
    }
    uint32_t idx = static_cast<uint32_t>(m_buffer_decls.size());
    m_buffer_decls.emplace_back();
    m_compiled_buffers.emplace_back();
    auto& decl = m_buffer_decls[idx];
    decl.debug_label = std::string(debug_label);
    decl.lifetime = Lifetime::Persistent;
    decl.active = true;
    decl.last_active_frame = m_frame_number;
    decl.external_buffer = buf;
    decl.external_size = size;
    decl.external_version = external_version;
    m_buffer_name_to_handle.emplace(std::string(debug_label), idx);
    return BufferDeclHandle{idx};
}

void FrameGraph::import_buffer(BufferDeclHandle h, WGPUBuffer buf, std::size_t size,
                               uint64_t external_version) {
    PTS_ZONE_SCOPED;
    PRECONDITION_MSG(buf != nullptr, "import_buffer: buffer must not be null");
    auto& decl = buf_decl(h);
    decl.active = true;
    decl.last_active_frame = m_frame_number;
    decl.external_buffer = buf;
    decl.external_size = size;
    decl.external_version = external_version;
}

void FrameGraph::resize(BufferDeclHandle h, BufferDesc new_desc) {
    PTS_ZONE_SCOPED;
    auto& decl = buf_decl(h);
    decl.active = true;
    decl.last_active_frame = m_frame_number;
    if (new_desc.size > decl.desc.size) {
        decl.desc.size = new_desc.size;
    }
    decl.desc.usage = static_cast<WGPUBufferUsage>(decl.desc.usage | new_desc.usage);
}

BufferDeclHandle FrameGraph::find_buffer(std::string_view label) const {
    auto it = m_buffer_name_to_handle.find(label);
    if (it == m_buffer_name_to_handle.end()) return BufferDeclHandle{};
    if (!m_buffer_decls[it->second].active) return BufferDeclHandle{};
    return BufferDeclHandle{it->second};
}

bool FrameGraph::valid(BufferDeclHandle h) const {
    return h && h.value < m_buffer_decls.size() && m_buffer_decls[h.value].active;
}

DescriptorDeclHandle FrameGraph::find_descriptor(std::string_view name) const {
    auto it = m_descriptor_name_to_handle.find(name);
    if (it == m_descriptor_name_to_handle.end()) return DescriptorDeclHandle{};
    if (!m_descriptor_decls[it->second].active) return DescriptorDeclHandle{};
    return DescriptorDeclHandle{it->second};
}

bool FrameGraph::valid(DescriptorDeclHandle h) const {
    return h && h.value < m_descriptor_decls.size() && m_descriptor_decls[h.value].active;
}

DescriptorBuilder FrameGraph::descriptor(std::string_view name, WGPUBindGroupLayout layout) {
    PTS_ZONE_SCOPED;
    return DescriptorBuilder(*this, std::string(name), layout);
}

DescriptorBuilder FrameGraph::descriptor(const IPass* pass, WGPUBindGroupLayout layout,
                                         const char* label) {
    PTS_ZONE_SCOPED;
    return DescriptorBuilder(*this, make_pass_key(pass, label, ResourceKind::Descriptor), layout);
}

// ── Pass-based helpers ───────────────────────────────────────────────────

std::string FrameGraph::make_pass_key(const IPass* pass, const char* label, ResourceKind kind) {
    PRECONDITION_MSG(pass != nullptr, "make_pass_key: pass must not be null");
    auto pass_name = pass->name();
    if (label) {
        std::string key;
        key.reserve(pass_name.size() + 1 + std::char_traits<char>::length(label));
        key.append(pass_name);
        key.push_back('/');
        key.append(label);
        return key;
    }
    auto& counters = m_pass_counters[pass_name];
    uint32_t n;
    std::string_view kind_name;
    switch (kind) {
        case ResourceKind::Texture:
            n = counters.texture++;
            kind_name = "texture";
            break;
        case ResourceKind::Buffer:
            n = counters.buffer++;
            kind_name = "buffer";
            break;
        case ResourceKind::Descriptor:
            n = counters.descriptor++;
            kind_name = "descriptor";
            break;
    }
    std::string key;
    key.reserve(pass_name.size() + 1 + kind_name.size() + 4);
    key.append(pass_name);
    key.push_back('/');
    key.append(kind_name);
    key.push_back('_');
    key.append(std::to_string(n));
    return key;
}

TextureDeclHandle FrameGraph::texture(const IPass* pass, TextureDesc desc, const char* label) {
    return texture(make_pass_key(pass, label, ResourceKind::Texture), desc);
}

BufferDeclHandle FrameGraph::buffer(const IPass* pass, BufferDesc desc, const char* label) {
    return buffer(make_pass_key(pass, label, ResourceKind::Buffer), desc);
}

BufferDeclHandle FrameGraph::import_buffer(const IPass* pass, WGPUBuffer buf, std::size_t size,
                                           uint64_t external_version, const char* label) {
    return import_buffer(make_pass_key(pass, label, ResourceKind::Buffer), buf, size,
                         external_version);
}

PassBuilder FrameGraph::add_pass(std::string name) {
    PTS_ZONE_SCOPED;
    Pass pass;
    pass.name = std::move(name);
    pass.index = static_cast<uint32_t>(m_passes.size());
    m_passes.push_back(std::move(pass));
    return PassBuilder(*this, static_cast<uint32_t>(m_passes.size() - 1));
}

// ── Frame lifecycle ──────────────────────────────────────────────────────

void FrameGraph::begin_frame() {
    PTS_ZONE_SCOPED;
    ++m_frame_number;
    m_passes.clear();
    m_pass_counters.clear();

    // Release old compiled resources deferred from the previous frame's compile().
    m_deferred_textures.clear();
    m_deferred_buffers.clear();

    // Reset per-frame scheduling state on active decls
    for (auto& decl : m_texture_decls) {
        if (!decl.active) continue;
        decl.first_writer = UINT32_MAX;
        decl.last_reader = UINT32_MAX;
    }
    for (auto& decl : m_buffer_decls) {
        if (!decl.active) continue;
        decl.first_writer = UINT32_MAX;
        decl.last_reader = UINT32_MAX;
    }
}

static bool descs_match(const TextureDesc& a, const TextureDesc& b) {
    return a.width == b.width && a.height == b.height && a.array_layers == b.array_layers &&
           a.format == b.format && a.usage == b.usage && a.force_array_view == b.force_array_view;
}

void FrameGraph::mark_liveness() {
    // Walk all passes and mark referenced decls as active this frame.
    auto mark_tex = [this](TextureDeclHandle h) {
        if (!h) return;
        auto& d = m_texture_decls[h.value];
        d.last_active_frame = m_frame_number;
    };

    auto mark_buf = [this](BufferDeclHandle h) {
        if (!h) return;
        auto& d = m_buffer_decls[h.value];
        d.last_active_frame = m_frame_number;
    };

    auto mark_desc = [&](DescriptorDeclHandle h) {
        if (!h) return;
        auto& d = m_descriptor_decls[h.value];
        d.last_active_frame = m_frame_number;
        // Transitively mark resources referenced by descriptor entries
        for (auto& entry : d.entries) {
            std::visit(
                [&](auto& b) {
                    using T = std::decay_t<decltype(b)>;
                    if constexpr (std::is_same_v<T, ManagedBufferBinding>) {
                        mark_buf(b.handle);
                    } else if constexpr (std::is_same_v<T, ManagedTextureBinding>) {
                        mark_tex(b.handle);
                    }
                },
                entry.resource);
        }
    };

    for (auto& pass : m_passes) {
        for (auto& att : pass.color_attachments) {
            mark_tex(att.handle);
        }
        if (pass.has_depth) {
            mark_tex(pass.depth_attachment.handle);
        }
        for (auto h : pass.reads) {
            mark_tex(h);
        }
        for (auto& slot : pass.descriptor_slots) {
            mark_desc(slot.handle);
        }
    }
}

void FrameGraph::compile() {
    PTS_ZONE_SCOPED;

    // Mark liveness from pass declarations
    mark_liveness();

    // Validate no backward dependencies (passes must be added in topological order)
    for (auto& pass : m_passes) {
        for (auto& att : pass.color_attachments) {
            if (!att.handle) continue;
            auto& decl = tex_decl(att.handle);
            if (att.is_read && decl.first_writer != UINT32_MAX && decl.first_writer > pass.index) {
                throw std::runtime_error("FrameGraph: backward dependency in pass '" + pass.name +
                                         "' reading resource '" + decl.debug_label +
                                         "' written by later pass");
            }
        }
        if (pass.has_depth && pass.depth_attachment.handle) {
            auto& decl = tex_decl(pass.depth_attachment.handle);
            if (pass.depth_attachment.is_read && decl.first_writer != UINT32_MAX &&
                decl.first_writer > pass.index) {
                throw std::runtime_error("FrameGraph: backward dependency in pass '" + pass.name +
                                         "' reading resource '" + decl.debug_label +
                                         "' written by later pass");
            }
        }
        for (auto h : pass.reads) {
            if (!h) continue;
            auto& decl = tex_decl(h);
            if (decl.first_writer != UINT32_MAX && decl.first_writer > pass.index) {
                throw std::runtime_error("FrameGraph: backward dependency in pass '" + pass.name +
                                         "' reading resource '" + decl.debug_label +
                                         "' written by later pass");
            }
        }
    }

    // Derive load/store ops
    for (auto& pass : m_passes) {
        if (pass.type == PassType::Compute) continue;

        for (auto& att : pass.color_attachments) {
            if (!att.handle) {
                // External view — always clear with provided clear color
                att.load_op = WGPULoadOp_Clear;
                att.store_op = WGPUStoreOp_Store;
                continue;
            }
            auto& decl = tex_decl(att.handle);
            if (att.layer != UINT32_MAX) {
                att.load_op = WGPULoadOp_Clear;
            } else if (att.is_write && decl.first_writer == pass.index) {
                att.load_op = WGPULoadOp_Clear;
            } else {
                att.load_op = WGPULoadOp_Load;
            }
            att.store_op = WGPUStoreOp_Store;
        }

        if (pass.has_depth) {
            auto& att = pass.depth_attachment;
            if (!att.handle) {
                // External depth view
                pass.depth_load_op = WGPULoadOp_Clear;
                pass.depth_store_op = WGPUStoreOp_Store;
            } else {
                auto& decl = tex_decl(att.handle);
                if (att.is_read && !att.is_write) {
                    pass.depth_read_only = true;
                    pass.depth_load_op = WGPULoadOp_Undefined;
                    pass.depth_store_op = WGPUStoreOp_Undefined;
                } else if (att.layer != UINT32_MAX) {
                    pass.depth_load_op = WGPULoadOp_Clear;
                    pass.depth_store_op = WGPUStoreOp_Store;
                } else if (att.is_write && decl.first_writer == pass.index) {
                    pass.depth_load_op = WGPULoadOp_Clear;
                    pass.depth_store_op = WGPUStoreOp_Store;
                } else {
                    pass.depth_load_op = WGPULoadOp_Load;
                    pass.depth_store_op = WGPUStoreOp_Store;
                }
            }
        }
    }

    materialize_textures();
    materialize_buffers();
    materialize_descriptors();
    evict_unused();
}

void FrameGraph::materialize_textures() {
    PTS_ZONE_SCOPED;
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_texture_decls.size()); ++i) {
        auto& decl = m_texture_decls[i];
        if (!decl.active) continue;

        if (decl.last_active_frame != m_frame_number) {
            if (decl.lifetime != Lifetime::Persistent) {
                decl.compiled = nullptr;
            }
            continue;
        }

        // Persistent with upload — create once, reuse forever
        if (decl.has_upload) {
            if (m_compiled_textures[i]) {
                decl.compiled = m_compiled_textures[i].get();
                continue;
            }
            auto tex = wgpuDeviceCreateTexture(m_device.handle(), &decl.upload_desc);
            INVARIANT_MSG(tex, "FrameGraph: failed to create persistent texture");

            WGPUTexelCopyBufferLayout layout = {};
            layout.bytesPerRow = decl.upload_bytes_per_row;
            layout.rowsPerImage = decl.upload_desc.size.height;
            WGPUTexelCopyTextureInfo dest = {};
            dest.texture = tex;
            dest.aspect = WGPUTextureAspect_All;
            WGPUExtent3D extent = decl.upload_desc.size;
            wgpuQueueWriteTexture(m_device.queue(), &dest, decl.upload_data,
                                  static_cast<size_t>(decl.upload_size), &layout, &extent);

            WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
            view_desc.format = decl.upload_desc.format;
            view_desc.dimension = decl.upload_view_dim;
            view_desc.mipLevelCount = decl.upload_desc.mipLevelCount;
            view_desc.arrayLayerCount = decl.upload_desc.size.depthOrArrayLayers;
            auto view = wgpuTextureCreateView(tex, &view_desc);
            INVARIANT_MSG(view, "FrameGraph: failed to create persistent texture view");

            auto compiled = std::make_unique<Texture>();
            compiled->texture = tex;
            compiled->view = view;
            compiled->desc = decl.desc;
            compiled->version = next_version();
            decl.compiled = compiled.get();
            m_compiled_textures[i] = std::move(compiled);
            continue;
        }

        // External view — no compiled backing.
        if (decl.external_view) {
            decl.compiled = nullptr;
            continue;
        }

        // Managed path — allocate or reuse based on desc match
        if (m_compiled_textures[i] && descs_match(m_compiled_textures[i]->desc, decl.desc)) {
            decl.compiled = m_compiled_textures[i].get();
            continue;
        }
        if (m_compiled_textures[i]) {
            m_logger->debug("FrameGraph: recreating texture '{}' (desc changed)", decl.debug_label);
            m_deferred_textures.push_back(std::move(m_compiled_textures[i]));
        }

        const uint32_t layers = decl.desc.array_layers;
        WGPUTextureDescriptor tex_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        tex_desc.label = {decl.debug_label.c_str(), decl.debug_label.size()};
        tex_desc.size = {decl.desc.width, decl.desc.height, layers};
        tex_desc.format = decl.desc.format;
        tex_desc.usage = decl.desc.usage;
        tex_desc.mipLevelCount = 1;
        tex_desc.sampleCount = 1;
        tex_desc.dimension = WGPUTextureDimension_2D;
        WGPUTexture texture = wgpuDeviceCreateTexture(m_device.handle(), &tex_desc);

        WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
        view_desc.format = decl.desc.format;
        view_desc.mipLevelCount = 1;

        bool use_array_view = layers > 1 || decl.desc.force_array_view;
        if (use_array_view) {
            view_desc.dimension = WGPUTextureViewDimension_2DArray;
            view_desc.arrayLayerCount = layers;
        } else {
            view_desc.dimension = WGPUTextureViewDimension_2D;
            view_desc.arrayLayerCount = 1;
        }
        WGPUTextureView view = wgpuTextureCreateView(texture, &view_desc);

        auto compiled = std::make_unique<Texture>();
        compiled->texture = texture;
        compiled->view = view;
        compiled->desc = decl.desc;
        compiled->version = next_version();

        if (use_array_view) {
            compiled->layer_views.reserve(layers);
            for (uint32_t l = 0; l < layers; ++l) {
                WGPUTextureViewDescriptor layer_view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
                layer_view_desc.format = decl.desc.format;
                layer_view_desc.dimension = WGPUTextureViewDimension_2D;
                layer_view_desc.mipLevelCount = 1;
                layer_view_desc.baseArrayLayer = l;
                layer_view_desc.arrayLayerCount = 1;
                compiled->layer_views.push_back(wgpuTextureCreateView(texture, &layer_view_desc));
            }
        }

        decl.compiled = compiled.get();
        m_compiled_textures[i] = std::move(compiled);

        m_logger->debug("FrameGraph: created texture '{}' ({}x{}, {} layers)", decl.debug_label,
                        decl.desc.width, decl.desc.height, layers);
    }
}

void FrameGraph::materialize_buffers() {
    PTS_ZONE_SCOPED;
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_buffer_decls.size()); ++i) {
        auto& decl = m_buffer_decls[i];
        if (!decl.active) continue;

        if (decl.last_active_frame != m_frame_number) {
            if (decl.lifetime != Lifetime::Persistent) {
                decl.compiled = nullptr;
            }
            continue;
        }

        // Imported buffer (external). Identity is (handle, external_version)
        // — same handle with a bumped version triggers a rebuild so descriptors
        // binding this buffer see a changed dep and rebuild their bind groups.
        if (decl.external_buffer) {
            if (m_compiled_buffers[i] && m_compiled_buffers[i]->buffer == decl.external_buffer &&
                m_compiled_buffers[i]->version == decl.external_version) {
                decl.compiled = m_compiled_buffers[i].get();
                continue;
            }
            if (m_compiled_buffers[i]) {
                m_compiled_buffers[i].reset();
            }
            auto compiled = std::make_unique<Buffer>();
            compiled->buffer = decl.external_buffer;
            compiled->size = decl.external_size;
            compiled->usage = WGPUBufferUsage_None;
            compiled->owned = false;
            // Buffer::version carries the caller-provided external_version so
            // descriptor cache deps detect external mutation without needing
            // the handle to change.
            compiled->version = decl.external_version != 0 ? decl.external_version : next_version();
            decl.compiled = compiled.get();
            auto final_version = compiled->version;
            m_compiled_buffers[i] = std::move(compiled);
            m_logger->debug("FrameGraph: imported buffer '{}' (size={}, v={})", decl.debug_label,
                            decl.external_size, final_version);
            continue;
        }

        // Persistent with initial upload
        if (decl.has_upload) {
            if (m_compiled_buffers[i]) {
                decl.compiled = m_compiled_buffers[i].get();
                continue;
            }
            WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
            buf_desc.label = {decl.debug_label.c_str(), decl.debug_label.size()};
            buf_desc.size = decl.desc.size;
            buf_desc.usage = decl.desc.usage;
            WGPUBuffer buffer = wgpuDeviceCreateBuffer(m_device.handle(), &buf_desc);
            INVARIANT_MSG(buffer, "FrameGraph: failed to create persistent buffer");
            wgpuQueueWriteBuffer(m_device.queue(), buffer, 0, decl.upload_data, decl.upload_size);

            auto compiled = std::make_unique<Buffer>();
            compiled->buffer = buffer;
            compiled->size = decl.desc.size;
            compiled->usage = decl.desc.usage;
            compiled->owned = true;
            compiled->version = next_version();
            decl.compiled = compiled.get();
            m_compiled_buffers[i] = std::move(compiled);
            m_logger->debug("FrameGraph: created persistent buffer '{}' (size={})",
                            decl.debug_label, decl.desc.size);
            continue;
        }

        // Managed buffer — reuse if sufficient size + superset usage
        if (m_compiled_buffers[i] && m_compiled_buffers[i]->size >= decl.desc.size &&
            (m_compiled_buffers[i]->usage & decl.desc.usage) == decl.desc.usage) {
            decl.compiled = m_compiled_buffers[i].get();
            continue;
        }
        if (m_compiled_buffers[i]) {
            m_deferred_buffers.push_back(std::move(m_compiled_buffers[i]));
        }

        WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        buf_desc.label = {decl.debug_label.c_str(), decl.debug_label.size()};
        buf_desc.size = decl.desc.size;
        buf_desc.usage = decl.desc.usage;
        WGPUBuffer buffer = wgpuDeviceCreateBuffer(m_device.handle(), &buf_desc);

        auto compiled = std::make_unique<Buffer>();
        compiled->buffer = buffer;
        compiled->size = decl.desc.size;
        compiled->usage = decl.desc.usage;
        compiled->owned = true;
        compiled->version = next_version();
        decl.compiled = compiled.get();
        m_compiled_buffers[i] = std::move(compiled);

        m_logger->debug("FrameGraph: created buffer '{}' (size={})", decl.debug_label,
                        decl.desc.size);
    }
}

void FrameGraph::materialize_descriptors() {
    PTS_ZONE_SCOPED;
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_descriptor_decls.size()); ++i) {
        auto& decl = m_descriptor_decls[i];
        if (!decl.active) continue;

        if (decl.last_active_frame != m_frame_number) {
            decl.compiled = nullptr;
            continue;
        }

        // Deps: BGL version + every bound resource's version (buffer/texture
        // compiled::version for managed, address for external). Descriptors
        // binding imported world buffers will rebuild when the caller-provided
        // external_version changes (propagated via Buffer::version).
        boost::container::small_vector<uint64_t, 8> deps;
        deps.push_back(bgl_version(decl.layout));
        for (auto& entry : decl.entries) {
            deps.push_back(std::visit(
                [&](auto& b) -> uint64_t {
                    using T = std::decay_t<decltype(b)>;
                    if constexpr (std::is_same_v<T, ManagedBufferBinding>) {
                        auto& bd = buf_decl(b.handle);
                        INVARIANT_MSG(bd.compiled != nullptr,
                                      "materialize_descriptors: buffer not compiled");
                        return bd.compiled->version;
                    } else if constexpr (std::is_same_v<T, ManagedTextureBinding>) {
                        auto& td = tex_decl(b.handle);
                        INVARIANT_MSG(td.compiled != nullptr,
                                      "materialize_descriptors: texture not compiled");
                        return td.compiled->version;
                    } else if constexpr (std::is_same_v<T, ExternalViewBinding>) {
                        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(b.view));
                    } else if constexpr (std::is_same_v<T, ExternalBufferBinding>) {
                        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(b.buffer));
                    } else if constexpr (std::is_same_v<T, SamplerBinding>) {
                        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(b.sampler));
                    }
                    return 0;
                },
                entry.resource));
        }

        const auto& ptr = m_descriptor_cache.get_or_build_with_replace(
            i, DescriptorCache::Span{deps.data(), deps.size()},
            [&]() -> std::unique_ptr<Descriptor> {
                std::vector<WGPUBindGroupEntry> wgpu_entries;
                wgpu_entries.reserve(decl.entries.size());
                for (auto& entry : decl.entries) {
                    WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
                    e.binding = entry.binding;
                    std::visit(
                        [&](auto& b) {
                            using T = std::decay_t<decltype(b)>;
                            if constexpr (std::is_same_v<T, ManagedBufferBinding>) {
                                auto* buf = buf_decl(b.handle).compiled;
                                e.buffer = buf->buffer;
                                e.offset = b.offset;
                                e.size = b.size > 0 ? b.size : buf->size;
                            } else if constexpr (std::is_same_v<T, ManagedTextureBinding>) {
                                auto* tex = tex_decl(b.handle).compiled;
                                if (b.layer != UINT32_MAX) {
                                    INVARIANT_MSG(
                                        b.layer < tex->layer_views.size(),
                                        "materialize_descriptors: texture layer out of range");
                                    e.textureView = tex->layer_views[b.layer];
                                } else {
                                    e.textureView = tex->view;
                                }
                            } else if constexpr (std::is_same_v<T, ExternalViewBinding>) {
                                e.textureView = b.view;
                            } else if constexpr (std::is_same_v<T, ExternalBufferBinding>) {
                                e.buffer = b.buffer;
                                e.offset = b.offset;
                                e.size = b.size;
                            } else if constexpr (std::is_same_v<T, SamplerBinding>) {
                                e.sampler = b.sampler;
                            }
                        },
                        entry.resource);
                    wgpu_entries.push_back(e);
                }

                WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
                bg_desc.label = {decl.debug_label.c_str(), decl.debug_label.size()};
                bg_desc.layout = decl.layout;
                bg_desc.entryCount = wgpu_entries.size();
                bg_desc.entries = wgpu_entries.data();
                WGPUBindGroup bg = wgpuDeviceCreateBindGroup(m_device.handle(), &bg_desc);

                auto compiled = std::make_unique<Descriptor>();
                compiled->bind_group = bg;
                compiled->version = next_version();
                m_logger->debug("FrameGraph: created descriptor '{}' (v{})", decl.debug_label,
                                compiled->version);
                return compiled;
            },
            [&](std::unique_ptr<Descriptor>& old) {
                m_logger->debug("FrameGraph: rebuilding descriptor '{}' (inputs changed)",
                                decl.debug_label);
                old.reset();
            });
        decl.compiled = ptr.get();
    }
}

void FrameGraph::evict_unused() {
    PTS_ZONE_SCOPED;
    // Descriptors: mark inactive, clear compiled. Bind groups are internal
    // to the FG so immediate destruction is safe.
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_descriptor_decls.size()); ++i) {
        auto& decl = m_descriptor_decls[i];
        if (!decl.active) continue;
        if (decl.last_active_frame == m_frame_number) continue;
        m_logger->debug("FrameGraph: evicting unused descriptor '{}'", decl.debug_label);
        m_descriptor_cache.erase(i);
        decl.compiled = nullptr;
        decl.active = false;
    }
    // Textures: mark inactive, defer compiled destruction
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_texture_decls.size()); ++i) {
        auto& decl = m_texture_decls[i];
        if (!decl.active) continue;
        if (decl.lifetime == Lifetime::Persistent) continue;
        if (decl.last_active_frame == m_frame_number) continue;
        m_logger->debug("FrameGraph: evicting unused texture '{}'", decl.debug_label);
        if (m_compiled_textures[i]) {
            m_deferred_textures.push_back(std::move(m_compiled_textures[i]));
        }
        decl.compiled = nullptr;
        decl.active = false;
    }
    // Buffers: mark inactive, defer compiled destruction
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_buffer_decls.size()); ++i) {
        auto& decl = m_buffer_decls[i];
        if (!decl.active) continue;
        if (decl.lifetime == Lifetime::Persistent) continue;
        if (decl.last_active_frame == m_frame_number) continue;
        m_logger->debug("FrameGraph: evicting unused buffer '{}'", decl.debug_label);
        if (m_compiled_buffers[i]) {
            m_deferred_buffers.push_back(std::move(m_compiled_buffers[i]));
        }
        decl.compiled = nullptr;
        decl.active = false;
    }
}

WGPUTextureView FrameGraph::resolve_view(const ColorAttachmentInfo& att) const {
    if (att.external_view) return att.external_view;
    if (!att.handle) return nullptr;
    auto& decl = tex_decl(att.handle);
    INVARIANT_MSG(decl.compiled, "resolve_view: color decl not compiled");
    if (att.layer != UINT32_MAX) {
        INVARIANT_MSG(att.layer < decl.compiled->layer_views.size(),
                      "resolve_view: layer out of range");
        return decl.compiled->layer_views[att.layer];
    }
    return decl.compiled->view;
}

WGPUTextureView FrameGraph::resolve_view(const DepthAttachmentInfo& att) const {
    if (att.external_view) return att.external_view;
    if (!att.handle) return nullptr;
    auto& decl = tex_decl(att.handle);
    INVARIANT_MSG(decl.compiled, "resolve_view: depth decl not compiled");
    if (att.layer != UINT32_MAX) {
        INVARIANT_MSG(att.layer < decl.compiled->layer_views.size(),
                      "resolve_view: layer out of range");
        return decl.compiled->layer_views[att.layer];
    }
    return decl.compiled->view;
}

void FrameGraph::execute(WGPUCommandEncoder encoder) {
    PTS_ZONE_SCOPED;
    for (auto& pass : m_passes) {
        ExecuteContext ctx{*this, m_frame_number};
        if (pass.type == PassType::Compute) {
            WGPUComputePassDescriptor desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
            desc.label = {pass.name.c_str(), pass.name.size()};
            auto enc = wgpuCommandEncoderBeginComputePass(encoder, &desc);
            for (auto& slot : pass.descriptor_slots) {
                if (slot.is_dynamic) continue;
                auto& dd = desc_decl(slot.handle);
                INVARIANT_MSG(dd.compiled, "static descriptor not compiled");
                wgpuComputePassEncoderSetBindGroup(enc, slot.index, dd.compiled->bind_group, 0,
                                                   nullptr);
            }
            if (pass.compute_fn) pass.compute_fn(ctx, enc);
            wgpuComputePassEncoderEnd(enc);
            wgpuComputePassEncoderRelease(enc);
        } else {
            std::vector<WGPURenderPassColorAttachment> color_attachments;
            color_attachments.reserve(pass.color_attachments.size());

            for (auto& att : pass.color_attachments) {
                WGPURenderPassColorAttachment color_attachment =
                    WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
                color_attachment.view = resolve_view(att);
                color_attachment.loadOp = att.load_op;
                color_attachment.storeOp = att.store_op;
                color_attachment.clearValue =
                    att.handle ? tex_decl(att.handle).desc.clear_color : att.external_clear;
                color_attachments.push_back(color_attachment);
            }

            WGPURenderPassDepthStencilAttachment depth_attachment =
                WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
            if (pass.has_depth) {
                auto& att = pass.depth_attachment;
                depth_attachment.view = resolve_view(att);
                depth_attachment.depthLoadOp = pass.depth_load_op;
                depth_attachment.depthStoreOp = pass.depth_store_op;
                depth_attachment.depthClearValue = att.handle
                                                       ? tex_decl(att.handle).desc.depth_clear_value
                                                       : att.external_clear_value;
                depth_attachment.depthReadOnly = pass.depth_read_only;
            }

            WGPURenderPassDescriptor pass_desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
            pass_desc.label = {pass.name.c_str(), pass.name.size()};
            if (!color_attachments.empty()) {
                pass_desc.colorAttachmentCount = color_attachments.size();
                pass_desc.colorAttachments = color_attachments.data();
            }
            if (pass.has_depth) {
                pass_desc.depthStencilAttachment = &depth_attachment;
            }

            WGPURenderPassEncoder pass_encoder =
                wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
            for (auto& slot : pass.descriptor_slots) {
                if (slot.is_dynamic) continue;
                auto& dd = desc_decl(slot.handle);
                INVARIANT_MSG(dd.compiled, "static descriptor not compiled");
                wgpuRenderPassEncoderSetBindGroup(pass_encoder, slot.index, dd.compiled->bind_group,
                                                  0, nullptr);
            }
            if (pass.render_fn) {
                pass.render_fn(ctx, pass_encoder);
            }
            wgpuRenderPassEncoderEnd(pass_encoder);
            wgpuRenderPassEncoderRelease(pass_encoder);
        }
    }
}

// ── Introspection ────────────────────────────────────────────────────────

size_t FrameGraph::cached_texture_count() const {
    size_t count = 0;
    for (auto& ptr : m_compiled_textures) {
        if (ptr) ++count;
    }
    return count;
}

size_t FrameGraph::cached_buffer_count() const {
    size_t count = 0;
    for (auto& ptr : m_compiled_buffers) {
        if (ptr) ++count;
    }
    return count;
}

size_t FrameGraph::cached_descriptor_count() const {
    size_t count = 0;
    m_descriptor_cache.for_each([&](uint32_t, const std::unique_ptr<Descriptor>& ptr) {
        if (ptr) ++count;
    });
    return count;
}

}  // namespace pts::rendering
