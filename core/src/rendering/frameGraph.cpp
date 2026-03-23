#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/webgpu/device.h>
#include <spdlog/spdlog.h>

#include <stdexcept>

namespace pts::rendering {

// --- CachedTexture ---

detail::CachedTexture::~CachedTexture() {
    if (view) {
        wgpuTextureViewRelease(view);
    }
    if (texture) {
        wgpuTextureRelease(texture);
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
    pass.color_attachments.push_back({h, false, true});
    return *this;
}

PassBuilder& PassBuilder::depth(ResourceHandle h) {
    auto& pass = m_graph.m_passes[m_pass_index];
    auto& res = m_graph.m_resources[h.index];
    if (res.first_writer == UINT32_MAX) {
        res.first_writer = m_pass_index;
    }
    pass.depth_attachment = {h, true, true};
    pass.has_depth = true;
    return *this;
}

PassBuilder& PassBuilder::depth_readonly(ResourceHandle h) {
    auto& pass = m_graph.m_passes[m_pass_index];
    pass.depth_attachment = {h, true, false};
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

void FrameGraph::begin_frame() {
    m_resources.clear();
    m_passes.clear();
    for (auto& [name, cached] : m_texture_cache) {
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
            if (att.is_write && res.first_writer == pass.index) {
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

    // Evict unused cached textures
    evict_unused();
}

static bool descs_match(const TextureDesc& a, const TextureDesc& b) {
    return a.width == b.width && a.height == b.height && a.format == b.format && a.usage == b.usage;
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

        // Overwrite cache entry if desc changed (old intrusive_ptr released automatically)
        if (it != m_texture_cache.end()) {
            m_logger->debug("FrameGraph: recreating texture '{}' (desc changed)", res.name);
            m_texture_cache.erase(it);
        }

        // Create new texture
        WGPUTextureDescriptor tex_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        tex_desc.size = {res.desc.width, res.desc.height, 1};
        tex_desc.format = res.desc.format;
        tex_desc.usage = res.desc.usage;
        tex_desc.mipLevelCount = 1;
        tex_desc.sampleCount = 1;
        tex_desc.dimension = WGPUTextureDimension_2D;
        WGPUTexture texture = wgpuDeviceCreateTexture(m_device.handle(), &tex_desc);

        WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
        view_desc.format = res.desc.format;
        view_desc.dimension = WGPUTextureViewDimension_2D;
        view_desc.mipLevelCount = 1;
        view_desc.arrayLayerCount = 1;
        WGPUTextureView view = wgpuTextureCreateView(texture, &view_desc);

        auto cached = boost::intrusive_ptr<detail::CachedTexture>(new detail::CachedTexture());
        cached->texture = texture;
        cached->view = view;
        cached->desc = res.desc;
        cached->used_this_frame = true;
        m_texture_cache[res.name] = cached;

        m_logger->debug("FrameGraph: created texture '{}' ({}x{})", res.name, res.desc.width,
                        res.desc.height);
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
                color_attachment.view = resolve_view(att.handle);
                color_attachment.loadOp = att.load_op;
                color_attachment.storeOp = att.store_op;
                color_attachment.clearValue = m_resources[att.handle.index].desc.clear_color;
                color_attachments.push_back(color_attachment);
            }

            WGPURenderPassDepthStencilAttachment depth_attachment =
                WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
            if (pass.has_depth) {
                auto& att = pass.depth_attachment;
                depth_attachment.view = resolve_view(att.handle);
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
