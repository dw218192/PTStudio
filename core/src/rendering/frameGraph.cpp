#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/device.h>
#include <spdlog/spdlog.h>

#include <stdexcept>

namespace pts::rendering {

// --- CachedTexture ---

detail::CachedTexture::~CachedTexture() {
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

// --- CachedBuffer ---

detail::CachedBuffer::~CachedBuffer() {
    if (owned && buffer) {
        wgpuBufferDestroy(buffer);
        wgpuBufferRelease(buffer);
    }
}

// --- CachedBindGroup ---

detail::CachedBindGroup::~CachedBindGroup() {
    if (bind_group) {
        wgpuBindGroupRelease(bind_group);
    }
}

// --- PassBuilder ---

PassBuilder::PassBuilder(FrameGraph& graph, uint32_t pass_index)
    : m_graph(graph), m_pass_index(pass_index) {
}

PassBuilder& PassBuilder::color(ResourceHandle h) {
    auto& pass = m_graph.m_passes[m_pass_index];
    auto& res = m_graph.m_resources[h.index];
    if (res.first_writer == UINT32_MAX) {
        res.first_writer = m_pass_index;
    }
    // Auto-infer RenderAttachment usage for managed resources
    if (!res.external_view) {
        res.desc.usage =
            static_cast<WGPUTextureUsage>(res.desc.usage | WGPUTextureUsage_RenderAttachment);
    }
    pass.color_attachments.push_back({h, UINT32_MAX, false, true});
    return *this;
}

PassBuilder& PassBuilder::color(ResourceHandle h, uint32_t layer) {
    auto& pass = m_graph.m_passes[m_pass_index];
    auto& res = m_graph.m_resources[h.index];
    PRECONDITION_MSG(res.desc.array_layers > 1 || res.desc.force_array_view,
                     "color(handle, layer) requires an array texture");
    PRECONDITION_MSG(layer < res.desc.array_layers, "layer index out of range");
    if (res.first_writer == UINT32_MAX) {
        res.first_writer = m_pass_index;
    }
    if (!res.external_view) {
        res.desc.usage =
            static_cast<WGPUTextureUsage>(res.desc.usage | WGPUTextureUsage_RenderAttachment);
    }
    pass.color_attachments.push_back({h, layer, false, true});
    return *this;
}

PassBuilder& PassBuilder::depth(ResourceHandle h) {
    auto& pass = m_graph.m_passes[m_pass_index];
    auto& res = m_graph.m_resources[h.index];
    if (res.first_writer == UINT32_MAX) {
        res.first_writer = m_pass_index;
    }
    pass.depth_attachment = {h, UINT32_MAX, true, true};
    pass.has_depth = true;
    return *this;
}

PassBuilder& PassBuilder::depth(ResourceHandle h, uint32_t layer) {
    auto& pass = m_graph.m_passes[m_pass_index];
    auto& res = m_graph.m_resources[h.index];
    PRECONDITION_MSG(res.desc.array_layers > 1 || res.desc.force_array_view,
                     "depth(handle, layer) requires an array texture");
    PRECONDITION_MSG(layer < res.desc.array_layers, "layer index out of range");
    if (res.first_writer == UINT32_MAX) {
        res.first_writer = m_pass_index;
    }
    pass.depth_attachment = {h, layer, true, true};
    pass.has_depth = true;
    return *this;
}

PassBuilder& PassBuilder::depth_readonly(ResourceHandle h) {
    auto& pass = m_graph.m_passes[m_pass_index];
    pass.depth_attachment = {h, UINT32_MAX, true, false};
    pass.has_depth = true;
    return *this;
}

PassBuilder& PassBuilder::color(WGPUTextureView view, WGPUColor clear_color) {
    // Dedup by pointer identity
    for (uint32_t i = 0; i < m_graph.m_resources.size(); ++i) {
        if (m_graph.m_resources[i].external_view == view) {
            return color(ResourceHandle{i});
        }
    }
    ResourceHandle h;
    h.index = static_cast<uint32_t>(m_graph.m_resources.size());
    FrameGraph::Resource res;
    res.desc.clear_color = clear_color;
    res.external_view = view;
    m_graph.m_resources.push_back(std::move(res));
    return color(h);
}

PassBuilder& PassBuilder::depth(WGPUTextureView view, float clear_value) {
    // Dedup by pointer identity
    for (uint32_t i = 0; i < m_graph.m_resources.size(); ++i) {
        if (m_graph.m_resources[i].external_view == view) {
            return depth(ResourceHandle{i});
        }
    }
    ResourceHandle h;
    h.index = static_cast<uint32_t>(m_graph.m_resources.size());
    FrameGraph::Resource res;
    res.desc.depth_clear_value = clear_value;
    res.external_view = view;
    m_graph.m_resources.push_back(std::move(res));
    return depth(h);
}

PassBuilder& PassBuilder::present() {
    m_graph.m_passes[m_pass_index].is_present = true;
    return *this;
}

PassBuilder& PassBuilder::read(ResourceHandle h) {
    auto& pass = m_graph.m_passes[m_pass_index];
    pass.reads.push_back(h);
    // Auto-infer TextureBinding usage for managed resources
    auto& res = m_graph.m_resources[h.index];
    if (!res.external_view) {
        res.desc.usage =
            static_cast<WGPUTextureUsage>(res.desc.usage | WGPUTextureUsage_TextureBinding);
    }
    return *this;
}

PassBuilder& PassBuilder::storage_write(ResourceHandle h) {
    auto& pass = m_graph.m_passes[m_pass_index];
    auto& res = m_graph.m_resources[h.index];
    if (res.first_writer == UINT32_MAX) {
        res.first_writer = m_pass_index;
    }
    // Auto-infer StorageBinding usage for managed resources
    if (!res.external_view) {
        res.desc.usage =
            static_cast<WGPUTextureUsage>(res.desc.usage | WGPUTextureUsage_StorageBinding);
    }
    pass.reads.push_back(h);
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

// --- FrameGraph ---

FrameGraph::FrameGraph(const webgpu::Device& device, std::shared_ptr<spdlog::logger> logger)
    : m_device(device), m_logger(std::move(logger)) {
}

FrameGraph::~FrameGraph() {
    m_bg_cache.clear();
    m_buffer_cache.clear();
    m_texture_cache.clear();
}

ResourceHandle FrameGraph::create(std::string name, TextureDesc desc) {
    for (auto& existing : m_resources) {
        PRECONDITION_MSG(existing.name != name,
                         "FrameGraph::create() called with duplicate resource name");
    }
    ResourceHandle h;
    h.index = static_cast<uint32_t>(m_resources.size());
    Resource res;
    res.name = std::move(name);
    res.desc = desc;
    m_resources.push_back(std::move(res));
    return h;
}

std::optional<ResourceHandle> FrameGraph::find(const std::string& name) const {
    for (uint32_t i = 0; i < m_resources.size(); ++i) {
        if (m_resources[i].name == name) {
            return ResourceHandle{i};
        }
    }
    return std::nullopt;
}

BufferHandle FrameGraph::find_or_create_buffer(std::string name, BufferDesc desc) {
    for (uint32_t i = 0; i < m_buffer_resources.size(); ++i) {
        if (m_buffer_resources[i].name == name) {
            auto& existing = m_buffer_resources[i];
            // Keep the larger size and merge usage flags
            if (desc.size > existing.desc.size) {
                existing.desc.size = desc.size;
            }
            existing.desc.usage = static_cast<WGPUBufferUsage>(existing.desc.usage | desc.usage);
            return BufferHandle{i};
        }
    }
    BufferResource res;
    res.name = std::move(name);
    res.desc = desc;
    BufferHandle h;
    h.index = static_cast<uint32_t>(m_buffer_resources.size());
    m_buffer_resources.push_back(std::move(res));
    return h;
}

BufferHandle FrameGraph::import_buffer(std::string name, WGPUBuffer buf, std::size_t size) {
    PRECONDITION_MSG(buf != nullptr, "import_buffer: buffer must not be null");
    for (uint32_t i = 0; i < m_buffer_resources.size(); ++i) {
        PRECONDITION_MSG(m_buffer_resources[i].name != name,
                         "import_buffer: duplicate buffer name");
    }
    BufferResource res;
    res.name = std::move(name);
    res.external_buffer = buf;
    res.external_size = size;
    BufferHandle h;
    h.index = static_cast<uint32_t>(m_buffer_resources.size());
    m_buffer_resources.push_back(std::move(res));
    return h;
}

std::optional<BufferHandle> FrameGraph::find_buffer(const std::string& name) const {
    for (uint32_t i = 0; i < m_buffer_resources.size(); ++i) {
        if (m_buffer_resources[i].name == name) {
            return BufferHandle{i};
        }
    }
    return std::nullopt;
}

BufferRef FrameGraph::get_buffer_ref(BufferHandle h) const {
    PRECONDITION_MSG(h.is_valid() && h.index < m_buffer_resources.size(),
                     "get_buffer_ref: invalid handle");
    BufferRef ref;
    auto& res = m_buffer_resources[h.index];
    auto it = m_buffer_cache.find(res.name);
    if (it != m_buffer_cache.end()) {
        ref.m_cached = it->second;
    }
    return ref;
}

BindGroupHandle FrameGraph::find_or_create_bind_group(std::string name, BindGroupDesc desc) {
    PRECONDITION_MSG(desc.layout != nullptr, "find_or_create_bind_group: layout must not be null");
    for (uint32_t i = 0; i < m_bg_resources.size(); ++i) {
        if (m_bg_resources[i].name == name) {
            return BindGroupHandle{i};
        }
    }
    BindGroupResource res;
    res.name = std::move(name);
    res.desc = std::move(desc);
    BindGroupHandle h;
    h.index = static_cast<uint32_t>(m_bg_resources.size());
    m_bg_resources.push_back(std::move(res));
    return h;
}

std::optional<BindGroupHandle> FrameGraph::find_bind_group(const std::string& name) const {
    for (uint32_t i = 0; i < m_bg_resources.size(); ++i) {
        if (m_bg_resources[i].name == name) {
            return BindGroupHandle{i};
        }
    }
    return std::nullopt;
}

BindGroupRef FrameGraph::get_bind_group_ref(BindGroupHandle h) const {
    PRECONDITION_MSG(h.is_valid() && h.index < m_bg_resources.size(),
                     "get_bind_group_ref: invalid handle");
    BindGroupRef ref;
    auto& res = m_bg_resources[h.index];
    auto it = m_bg_cache.find(res.name);
    if (it != m_bg_cache.end()) {
        ref.m_cached = it->second;
    }
    return ref;
}

ResourceHandle FrameGraph::find_or_create(std::string name, TextureDesc desc) {
    for (uint32_t i = 0; i < m_resources.size(); ++i) {
        if (m_resources[i].name == name) {
            auto& existing = m_resources[i];
            INVARIANT_MSG(existing.desc.format == desc.format,
                          "find_or_create: format mismatch for existing resource");
            INVARIANT_MSG(existing.desc.width == desc.width,
                          "find_or_create: width mismatch for existing resource");
            INVARIANT_MSG(existing.desc.height == desc.height,
                          "find_or_create: height mismatch for existing resource");
            INVARIANT_MSG(existing.desc.array_layers == desc.array_layers,
                          "find_or_create: array_layers mismatch for existing resource");
            // Merge usage flags — later consumers may need additional access (e.g. CopySrc)
            existing.desc.usage = static_cast<WGPUTextureUsage>(existing.desc.usage | desc.usage);
            return ResourceHandle{i};
        }
    }
    return create(std::move(name), desc);
}

PassBuilder FrameGraph::add_pass(std::string name) {
    Pass pass;
    pass.name = std::move(name);
    pass.index = static_cast<uint32_t>(m_passes.size());
    m_passes.push_back(std::move(pass));

    return PassBuilder(*this, static_cast<uint32_t>(m_passes.size() - 1));
}

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
        case ResourceKind::BindGroup:
            n = counters.bind_group++;
            kind_name = "bind_group";
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

ResourceHandle FrameGraph::find_or_create(const IPass* pass, TextureDesc desc, const char* label) {
    return find_or_create(make_pass_key(pass, label, ResourceKind::Texture), desc);
}

BufferHandle FrameGraph::find_or_create_buffer(const IPass* pass, BufferDesc desc,
                                               const char* label) {
    return find_or_create_buffer(make_pass_key(pass, label, ResourceKind::Buffer), desc);
}

BufferHandle FrameGraph::import_buffer(const IPass* pass, WGPUBuffer buf, std::size_t size,
                                       const char* label) {
    return import_buffer(make_pass_key(pass, label, ResourceKind::Buffer), buf, size);
}

BindGroupHandle FrameGraph::find_or_create_bind_group(const IPass* pass, BindGroupDesc desc,
                                                      const char* label) {
    return find_or_create_bind_group(make_pass_key(pass, label, ResourceKind::BindGroup),
                                     std::move(desc));
}

void FrameGraph::begin_frame() {
    m_resources.clear();
    m_passes.clear();
    m_buffer_resources.clear();
    m_bg_resources.clear();
    m_pass_counters.clear();
    for (auto& [name, cached] : m_texture_cache) {
        cached->used_this_frame = false;
    }
    for (auto& [name, cached] : m_buffer_cache) {
        cached->used_this_frame = false;
    }
    for (auto& [name, cached] : m_bg_cache) {
        cached->used_this_frame = false;
    }
}

void FrameGraph::compile() {
    PTS_ZONE_SCOPED;
    // Validate no backward dependencies (passes must be added in topological order)
    for (auto& pass : m_passes) {
        for (auto& att : pass.color_attachments) {
            if (!att.handle.is_valid()) continue;
            auto& res = m_resources[att.handle.index];
            if (att.is_read && res.first_writer != UINT32_MAX && res.first_writer > pass.index) {
                throw std::runtime_error("FrameGraph: backward dependency in pass '" + pass.name +
                                         "' reading resource '" + res.name +
                                         "' written by later pass");
            }
        }
        if (pass.has_depth && pass.depth_attachment.handle.is_valid()) {
            auto& res = m_resources[pass.depth_attachment.handle.index];
            if (pass.depth_attachment.is_read && res.first_writer != UINT32_MAX &&
                res.first_writer > pass.index) {
                throw std::runtime_error("FrameGraph: backward dependency in pass '" + pass.name +
                                         "' reading resource '" + res.name +
                                         "' written by later pass");
            }
        }
        for (auto& rh : pass.reads) {
            if (!rh.is_valid()) continue;
            auto& res = m_resources[rh.index];
            if (res.first_writer != UINT32_MAX && res.first_writer > pass.index) {
                throw std::runtime_error("FrameGraph: backward dependency in pass '" + pass.name +
                                         "' reading resource '" + res.name +
                                         "' written by later pass");
            }
        }
    }

    // Derive load/store ops
    for (auto& pass : m_passes) {
        // Skip load/store derivation for compute passes
        if (pass.type == PassType::Compute) {
            continue;
        }

        // Color attachments - per-attachment load/store ops (MRT)
        for (auto& att : pass.color_attachments) {
            auto& res = m_resources[att.handle.index];
            if (att.layer != UINT32_MAX) {
                // Layer-targeted: always clear — each layer is independent
                att.load_op = WGPULoadOp_Clear;
            } else if (att.is_write && res.first_writer == pass.index) {
                att.load_op = WGPULoadOp_Clear;
            } else {
                att.load_op = WGPULoadOp_Load;
            }
            att.store_op = WGPUStoreOp_Store;
        }

        // Depth attachment
        if (pass.has_depth) {
            auto& att = pass.depth_attachment;
            auto& res = m_resources[att.handle.index];
            if (att.is_read && !att.is_write) {
                // Read-only depth
                pass.depth_read_only = true;
                pass.depth_load_op = WGPULoadOp_Undefined;
                pass.depth_store_op = WGPUStoreOp_Undefined;
            } else if (att.layer != UINT32_MAX) {
                // Layer-targeted: always clear — each layer is independent
                pass.depth_load_op = WGPULoadOp_Clear;
                pass.depth_store_op = WGPUStoreOp_Store;
            } else if (att.is_write && res.first_writer == pass.index) {
                pass.depth_load_op = WGPULoadOp_Clear;
                pass.depth_store_op = WGPUStoreOp_Store;
            } else {
                pass.depth_load_op = WGPULoadOp_Load;
                pass.depth_store_op = WGPUStoreOp_Store;
            }
        }
    }

    // Allocate transient textures
    allocate_textures();

    // Allocate buffers
    allocate_buffers();

    // Allocate bind groups (after textures and buffers are resolved)
    allocate_bind_groups();

    // Evict unused cached resources
    evict_unused();
}

static bool descs_match(const TextureDesc& a, const TextureDesc& b) {
    return a.width == b.width && a.height == b.height && a.array_layers == b.array_layers &&
           a.format == b.format && a.usage == b.usage && a.force_array_view == b.force_array_view;
}

void FrameGraph::allocate_textures() {
    for (auto& res : m_resources) {
        if (res.external_view) continue;

        auto it = m_texture_cache.find(res.name);
        if (it != m_texture_cache.end() && descs_match(it->second->desc, res.desc)) {
            // Reuse cached texture
            it->second->used_this_frame = true;
            continue;
        }

        // Capture previous version before evicting stale entry
        uint64_t prev_version = 0;
        if (it != m_texture_cache.end()) {
            prev_version = it->second->version;
            m_logger->debug("FrameGraph: recreating texture '{}' (desc changed)", res.name);
            m_texture_cache.erase(it);
        }

        // Create new texture
        const uint32_t layers = res.desc.array_layers;
        WGPUTextureDescriptor tex_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        tex_desc.label = {res.name.c_str(), res.name.size()};
        tex_desc.size = {res.desc.width, res.desc.height, layers};
        tex_desc.format = res.desc.format;
        tex_desc.usage = res.desc.usage;
        tex_desc.mipLevelCount = 1;
        tex_desc.sampleCount = 1;
        tex_desc.dimension = WGPUTextureDimension_2D;
        WGPUTexture texture = wgpuDeviceCreateTexture(m_device.handle(), &tex_desc);

        WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
        view_desc.format = res.desc.format;
        view_desc.mipLevelCount = 1;

        bool use_array_view = layers > 1 || res.desc.force_array_view;
        if (use_array_view) {
            view_desc.dimension = WGPUTextureViewDimension_2DArray;
            view_desc.arrayLayerCount = layers;
        } else {
            view_desc.dimension = WGPUTextureViewDimension_2D;
            view_desc.arrayLayerCount = 1;
        }
        WGPUTextureView view = wgpuTextureCreateView(texture, &view_desc);

        auto cached = boost::intrusive_ptr<detail::CachedTexture>(new detail::CachedTexture());
        cached->texture = texture;
        cached->view = view;
        cached->desc = res.desc;
        cached->used_this_frame = true;
        cached->version = prev_version + 1;

        // Create per-layer views for array textures
        if (use_array_view) {
            cached->layer_views.reserve(layers);
            for (uint32_t i = 0; i < layers; ++i) {
                WGPUTextureViewDescriptor layer_view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
                layer_view_desc.format = res.desc.format;
                layer_view_desc.dimension = WGPUTextureViewDimension_2D;
                layer_view_desc.mipLevelCount = 1;
                layer_view_desc.baseArrayLayer = i;
                layer_view_desc.arrayLayerCount = 1;
                cached->layer_views.push_back(wgpuTextureCreateView(texture, &layer_view_desc));
            }
        }

        m_texture_cache[res.name] = cached;

        m_logger->debug("FrameGraph: created texture '{}' ({}x{}, {} layers)", res.name,
                        res.desc.width, res.desc.height, layers);
    }
}

void FrameGraph::allocate_buffers() {
    for (auto& res : m_buffer_resources) {
        if (res.external_buffer) {
            // Imported buffer
            auto it = m_buffer_cache.find(res.name);
            if (it != m_buffer_cache.end() && it->second->buffer == res.external_buffer) {
                // Same pointer — reuse, keep version
                it->second->used_this_frame = true;
                continue;
            }

            // Different pointer or new entry — bump version
            uint64_t prev_version = 0;
            if (it != m_buffer_cache.end()) {
                prev_version = it->second->version;
            }

            auto cached = boost::intrusive_ptr<detail::CachedBuffer>(new detail::CachedBuffer());
            cached->buffer = res.external_buffer;
            cached->desc.size = res.external_size;
            cached->desc.usage = WGPUBufferUsage_None;
            cached->owned = false;
            cached->used_this_frame = true;
            cached->version = prev_version + 1;
            m_buffer_cache[res.name] = cached;

            m_logger->debug("FrameGraph: imported buffer '{}' (size={})", res.name,
                            res.external_size);
        } else {
            // Managed buffer
            auto it = m_buffer_cache.find(res.name);
            if (it != m_buffer_cache.end() && it->second->desc.size >= res.desc.size &&
                (it->second->desc.usage & res.desc.usage) == res.desc.usage) {
                // Sufficient size and superset usage — reuse
                it->second->used_this_frame = true;
                continue;
            }

            // Need new buffer — capture previous version before evicting
            uint64_t prev_version = 0;
            if (it != m_buffer_cache.end()) {
                prev_version = it->second->version;
                m_buffer_cache.erase(it);
            }

            WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
            buf_desc.label = {res.name.c_str(), res.name.size()};
            buf_desc.size = res.desc.size;
            buf_desc.usage = res.desc.usage;
            WGPUBuffer buffer = wgpuDeviceCreateBuffer(m_device.handle(), &buf_desc);

            auto cached = boost::intrusive_ptr<detail::CachedBuffer>(new detail::CachedBuffer());
            cached->buffer = buffer;
            cached->desc = res.desc;
            cached->owned = true;
            cached->used_this_frame = true;
            cached->version = prev_version + 1;
            m_buffer_cache[res.name] = cached;

            m_logger->debug("FrameGraph: created buffer '{}' (size={})", res.name, res.desc.size);
        }
    }
}

void FrameGraph::allocate_bind_groups() {
    for (auto& res : m_bg_resources) {
        auto& desc = res.desc;

        // 1. Resolve current version for each entry
        std::vector<uint64_t> current_versions;
        current_versions.reserve(desc.entries.size());
        for (auto& entry : desc.entries) {
            if (entry.buffer.is_valid()) {
                INVARIANT_MSG(entry.buffer.index < m_buffer_resources.size(),
                              "allocate_bind_groups: buffer handle out of range");
                auto& buf_name = m_buffer_resources[entry.buffer.index].name;
                auto it = m_buffer_cache.find(buf_name);
                INVARIANT_MSG(it != m_buffer_cache.end(),
                              "allocate_bind_groups: buffer not in cache");
                current_versions.push_back(it->second->version);
            } else if (entry.texture.is_valid()) {
                INVARIANT_MSG(entry.texture.index < m_resources.size(),
                              "allocate_bind_groups: texture handle out of range");
                auto& tex_name = m_resources[entry.texture.index].name;
                auto it = m_texture_cache.find(tex_name);
                INVARIANT_MSG(it != m_texture_cache.end(),
                              "allocate_bind_groups: texture not in cache");
                current_versions.push_back(it->second->version);
            } else {
                // external_view, external_buffer, sampler — no version tracking
                current_versions.push_back(0);
            }
        }

        // 2. Check cache for version match
        auto cache_it = m_bg_cache.find(res.name);
        if (cache_it != m_bg_cache.end() &&
            cache_it->second->input_versions_snapshot == current_versions) {
            cache_it->second->used_this_frame = true;
            continue;
        }

        // 3. Versions differ or new entry — rebuild
        uint64_t prev_version = 0;
        if (cache_it != m_bg_cache.end()) {
            prev_version = cache_it->second->version;
            m_logger->debug("FrameGraph: rebuilding bind group '{}' (input versions changed)",
                            res.name);
        }

        // Build WGPUBindGroupEntry array from resolved resources
        std::vector<WGPUBindGroupEntry> wgpu_entries;
        wgpu_entries.reserve(desc.entries.size());
        for (auto& entry : desc.entries) {
            WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
            e.binding = entry.binding;

            if (entry.buffer.is_valid()) {
                auto& buf_name = m_buffer_resources[entry.buffer.index].name;
                auto& cached_buf = m_buffer_cache.at(buf_name);
                e.buffer = cached_buf->buffer;
                e.offset = entry.buffer_offset;
                e.size = entry.buffer_size > 0 ? entry.buffer_size : cached_buf->desc.size;
            } else if (entry.texture.is_valid()) {
                PRECONDITION_MSG(entry.texture_layer == UINT32_MAX,
                                 "allocate_bind_groups: texture_layer not yet supported");
                auto& tex_name = m_resources[entry.texture.index].name;
                auto& cached_tex = m_texture_cache.at(tex_name);
                e.textureView = cached_tex->view;
            } else if (entry.sampler) {
                e.sampler = entry.sampler;
            } else if (entry.external_view) {
                e.textureView = entry.external_view;
            } else if (entry.external_buffer) {
                e.buffer = entry.external_buffer;
                e.offset = entry.buffer_offset;
                e.size = entry.external_buffer_size;
            }

            wgpu_entries.push_back(e);
        }

        WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bg_desc.label = {res.name.c_str(), res.name.size()};
        bg_desc.layout = desc.layout;
        bg_desc.entryCount = wgpu_entries.size();
        bg_desc.entries = wgpu_entries.data();
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(m_device.handle(), &bg_desc);

        auto cached = boost::intrusive_ptr<detail::CachedBindGroup>(new detail::CachedBindGroup());
        cached->bind_group = bg;
        cached->input_versions_snapshot = std::move(current_versions);
        cached->used_this_frame = true;
        cached->version = prev_version + 1;
        m_bg_cache[res.name] = cached;

        m_logger->debug("FrameGraph: created bind group '{}' (v{})", res.name, cached->version);
    }
}

void FrameGraph::evict_unused() {
    for (auto it = m_texture_cache.begin(); it != m_texture_cache.end();) {
        if (!it->second->used_this_frame) {
            m_logger->debug("FrameGraph: evicting unused texture '{}'", it->first);
            it = m_texture_cache.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = m_buffer_cache.begin(); it != m_buffer_cache.end();) {
        if (!it->second->used_this_frame) {
            m_logger->debug("FrameGraph: evicting unused buffer '{}'", it->first);
            it = m_buffer_cache.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = m_bg_cache.begin(); it != m_bg_cache.end();) {
        if (!it->second->used_this_frame) {
            m_logger->debug("FrameGraph: evicting unused bind group '{}'", it->first);
            it = m_bg_cache.erase(it);
        } else {
            ++it;
        }
    }
}

TextureRef FrameGraph::get_texture_ref(ResourceHandle h) const {
    TextureRef ref;
    auto& res = m_resources[h.index];
    PRECONDITION_MSG(!res.external_view, "get_texture_ref() cannot be used on external resources");
    auto it = m_texture_cache.find(res.name);
    if (it != m_texture_cache.end()) {
        ref.m_cached = it->second;
    }
    return ref;
}

WGPUTextureView FrameGraph::resolve_view(ResourceHandle h) const {
    auto& res = m_resources[h.index];
    if (res.external_view) {
        return res.external_view;
    }
    auto it = m_texture_cache.find(res.name);
    if (it != m_texture_cache.end()) {
        return it->second->view;
    }
    return nullptr;
}

WGPUTextureView FrameGraph::resolve_layer_view(ResourceHandle h, uint32_t layer) const {
    auto& res = m_resources[h.index];
    PRECONDITION_MSG(!res.external_view,
                     "resolve_layer_view: not supported for external resources");
    auto it = m_texture_cache.find(res.name);
    PRECONDITION_MSG(it != m_texture_cache.end(), "resolve_layer_view: texture not allocated");
    PRECONDITION_MSG(layer < it->second->layer_views.size(),
                     "resolve_layer_view: layer out of range");
    return it->second->layer_views[layer];
}

void FrameGraph::execute(WGPUCommandEncoder encoder) {
    PTS_ZONE_SCOPED;
    for (auto& pass : m_passes) {
        if (pass.type == PassType::Compute) {
            WGPUComputePassDescriptor desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
            desc.label = {pass.name.c_str(), pass.name.size()};
            auto enc = wgpuCommandEncoderBeginComputePass(encoder, &desc);
            if (pass.compute_fn) pass.compute_fn(enc);
            wgpuComputePassEncoderEnd(enc);
            wgpuComputePassEncoderRelease(enc);
        } else {
            // Build MRT color attachment array
            std::vector<WGPURenderPassColorAttachment> color_attachments;
            color_attachments.reserve(pass.color_attachments.size());

            for (auto& att : pass.color_attachments) {
                WGPURenderPassColorAttachment color_attachment =
                    WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
                color_attachment.view = att.layer != UINT32_MAX
                                            ? resolve_layer_view(att.handle, att.layer)
                                            : resolve_view(att.handle);
                color_attachment.loadOp = att.load_op;
                color_attachment.storeOp = att.store_op;
                color_attachment.clearValue = m_resources[att.handle.index].desc.clear_color;
                color_attachments.push_back(color_attachment);
            }

            WGPURenderPassDepthStencilAttachment depth_attachment =
                WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
            if (pass.has_depth) {
                auto& att = pass.depth_attachment;
                depth_attachment.view = att.layer != UINT32_MAX
                                            ? resolve_layer_view(att.handle, att.layer)
                                            : resolve_view(att.handle);
                depth_attachment.depthLoadOp = pass.depth_load_op;
                depth_attachment.depthStoreOp = pass.depth_store_op;
                depth_attachment.depthClearValue =
                    m_resources[att.handle.index].desc.depth_clear_value;
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
            if (pass.render_fn) {
                pass.render_fn(pass_encoder);
            }
            wgpuRenderPassEncoderEnd(pass_encoder);
            wgpuRenderPassEncoderRelease(pass_encoder);
        }
    }
}

}  // namespace pts::rendering
