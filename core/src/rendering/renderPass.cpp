#include <core/rendering/passContext.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/renderer.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/toneMappingPass.h>
#include <core/rendering/webgpu/device.h>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>

namespace pts::rendering {

namespace {

constexpr const char* k_no_debug_define = "NO_DEBUG_TARGETS";

// WebGPU spec "color attachment bytes per sample" algorithm.
// See https://www.w3.org/TR/webgpu/#abstract-opdef-color-attachment-bytes-per-sample
//
// For each attachment:
//   total = roundUp(total, renderTargetComponentAlignment) + renderTargetPixelByteCost
//
// The per-format values come from the spec's format capability table.
// renderTargetPixelByteCost can be LARGER than the texel block size —
// e.g. RGBA8Unorm has a 4-byte texel block but costs 8 bytes as a render target.

struct RenderTargetCost {
    uint32_t pixel_byte_cost;
    uint32_t component_alignment;
};

RenderTargetCost render_target_cost(WGPUTextureFormat format) {
    switch (format) {
        // 1-component 8-bit
        case WGPUTextureFormat_R8Unorm:
        case WGPUTextureFormat_R8Snorm:
        case WGPUTextureFormat_R8Uint:
        case WGPUTextureFormat_R8Sint:
            return {1, 1};
        // 2-component 8-bit / 1-component 16-bit
        case WGPUTextureFormat_R16Uint:
        case WGPUTextureFormat_R16Sint:
        case WGPUTextureFormat_R16Float:
        case WGPUTextureFormat_RG8Unorm:
        case WGPUTextureFormat_RG8Snorm:
        case WGPUTextureFormat_RG8Uint:
        case WGPUTextureFormat_RG8Sint:
            return {2, 1};
        // 2-component 16-bit / 1-component 32-bit
        case WGPUTextureFormat_R32Float:
        case WGPUTextureFormat_R32Uint:
        case WGPUTextureFormat_R32Sint:
        case WGPUTextureFormat_RG16Uint:
        case WGPUTextureFormat_RG16Sint:
        case WGPUTextureFormat_RG16Float:
            return {4, 2};
        // 4-component 8-bit / 2-component 32-bit / 4-component 16-bit / packed 32-bit
        case WGPUTextureFormat_RGBA8Unorm:
        case WGPUTextureFormat_RGBA8UnormSrgb:
        case WGPUTextureFormat_RGBA8Snorm:
        case WGPUTextureFormat_RGBA8Uint:
        case WGPUTextureFormat_RGBA8Sint:
        case WGPUTextureFormat_BGRA8Unorm:
        case WGPUTextureFormat_BGRA8UnormSrgb:
        case WGPUTextureFormat_RGB10A2Uint:
        case WGPUTextureFormat_RGB10A2Unorm:
        case WGPUTextureFormat_RG11B10Ufloat:
        case WGPUTextureFormat_RG32Float:
        case WGPUTextureFormat_RG32Uint:
        case WGPUTextureFormat_RG32Sint:
        case WGPUTextureFormat_RGBA16Uint:
        case WGPUTextureFormat_RGBA16Sint:
        case WGPUTextureFormat_RGBA16Float:
            return {8, 4};
        // 4-component 32-bit
        case WGPUTextureFormat_RGBA32Float:
        case WGPUTextureFormat_RGBA32Uint:
        case WGPUTextureFormat_RGBA32Sint:
            return {16, 4};
        default:
            spdlog::warn("Unknown render target format {} — assuming 16 bytes",
                         static_cast<int>(format));
            return {16, 4};
    }
}

uint32_t round_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

uint32_t color_attachment_bytes_per_sample(WGPUTextureFormat scene_format,
                                           WGPUTextureFormat debug_format, uint32_t debug_count) {
    auto scene = render_target_cost(scene_format);
    uint32_t total = round_up(0, scene.component_alignment) + scene.pixel_byte_cost;

    auto debug = render_target_cost(debug_format);
    for (uint32_t i = 0; i < debug_count; ++i) {
        total = round_up(total, debug.component_alignment) + debug.pixel_byte_cost;
    }
    return total;
}

}  // namespace

IPass::IPass(const ShaderLoader& shader_loader) : m_shader_loader(&shader_loader) {
}

void IPass::ensure_initialized(const webgpu::Device& device) {
    if (m_initialized) return;
    m_initialized = true;

    // Create a per-pass logger sharing the ShaderLoader's sinks and level.
    // This mirrors LoggingManager::get_logger_shared — same sinks/pattern —
    // without requiring IPass to hold a LoggingManager reference.
    auto pass_name = std::string{name()};
    m_logger = spdlog::get(pass_name);
    if (!m_logger) {
        auto& parent = *m_shader_loader->logger();
        m_logger = std::make_shared<spdlog::logger>(pass_name, parent.sinks().begin(),
                                                    parent.sinks().end());
        m_logger->set_level(parent.level());
        spdlog::register_logger(m_logger);
    }
    compute_allowed_debug_targets(device);
}

void IPass::compute_allowed_debug_targets(const webgpu::Device& device) {
    auto [targets, desired] = debug_targets();
    if (desired == 0) {
        m_allowed_debug_count = 0;
        return;
    }

    WGPULimits limits = WGPU_LIMITS_INIT;
    wgpuDeviceGetLimits(device.handle(), &limits);

    // Binary-reduce: find the largest count that fits both attachment count
    // and bytes-per-sample limits. Formats are hardcoded to match the
    // convention: scene_color = RGBA16Float, debug targets = RGBA8Unorm.
    constexpr auto k_scene_fmt = WGPUTextureFormat_RGBA16Float;
    constexpr auto k_debug_fmt = WGPUTextureFormat_RGBA8Unorm;

    // All-or-nothing: either all debug targets fit, or none are used.
    // A no-debug shader variant is selected at pipeline creation time.
    uint32_t total_attachments = 1 + desired;
    uint32_t bytes = color_attachment_bytes_per_sample(k_scene_fmt, k_debug_fmt, desired);
    bool fits = total_attachments <= limits.maxColorAttachments &&
                bytes <= limits.maxColorAttachmentBytesPerSample;

    if (!fits) {
        spdlog::warn(
            "pass '{}': debug targets disabled: {} bytes/sample exceeds device limit {} "
            "(maxColorAttachments={})",
            name(), bytes, limits.maxColorAttachmentBytesPerSample, limits.maxColorAttachments);
    }

    m_allowed_debug_count = fits ? desired : 0;
}

auto IPass::load_pass_shader(std::string_view resource_key) const -> std::string {
    auto [targets, count] = effective_debug_targets();
    if (count > 0) {
        return m_shader_loader->load(resource_key);
    }
    // Derive variant key: "path/foo.wgsl" → "path/foo_no_debug.wgsl"
    auto key = std::string(resource_key);
    auto dot = key.rfind('.');
    INVARIANT_MSG(dot != std::string::npos, "resource_key must have an extension");
    auto variant_key = key.substr(0, dot) + "_no_debug" + key.substr(dot);
    return m_shader_loader->load(variant_key);
}

auto IPass::load_pass_shader_module(FrameGraph& fg, std::string_view resource_key) const
    -> WGPUShaderModule {
    auto [targets, count] = effective_debug_targets();
    if (count > 0) {
        return fg.shader(resource_key);
    }
    auto key = std::string(resource_key);
    auto dot = key.rfind('.');
    INVARIANT_MSG(dot != std::string::npos, "resource_key must have an extension");
    auto variant_key = key.substr(0, dot) + "_no_debug" + key.substr(dot);
    std::string_view defines[] = {k_no_debug_define};
    return fg.shader_variant(variant_key, resource_key, defines);
}

IRenderer::IRenderer(const ShaderLoader& shader_loader)
    : IPass(shader_loader), m_tonemapping(std::make_unique<ToneMappingPass>(shader_loader)) {
}

IRenderer::~IRenderer() = default;

void IRenderer::ensure_initialized(const webgpu::Device& device) {
    IPass::ensure_initialized(device);
    for (auto& c : m_children) c->ensure_initialized(device);
    m_tonemapping->ensure_initialized(device);
    // Collected once on first init. If children's effective_debug_targets
    // change after a device-limit re-query, clear m_all_debug_targets to
    // force a recollect on the next call.
    if (m_all_debug_targets.empty()) {
        collect_debug_targets();
    }
}

void IRenderer::collect_debug_targets() {
    m_all_debug_targets.clear();
    // Renderer's own targets (from derived class override of renderer_debug_targets)
    auto [own, own_count] = renderer_debug_targets();
    for (uint32_t i = 0; i < own_count; ++i) {
        m_all_debug_targets.push_back(own[i]);
    }
    // Children's targets
    for (auto& c : m_children) {
        auto [targets, count] = c->effective_debug_targets();
        for (uint32_t i = 0; i < count; ++i) {
            m_all_debug_targets.push_back(targets[i]);
        }
    }
}

IRenderer::Outputs IRenderer::add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) {
    ensure_initialized(ctx.device);
    auto hdr = do_add_to_frame_graph(fg, ctx);
    INVARIANT_MSG(hdr.color, "Renderer must produce a color output");

    // Run tone mapping on HDR color → LDR display-ready
    INVARIANT(m_tonemapping);
    TextureDeclHandle display_color = hdr.color;
    if (m_tonemapping_enabled) {
        m_tonemapping->set_inputs({hdr.color, hdr.depth, hdr.ssao});
        m_tonemapping->add_to_frame_graph(fg, ctx);
        display_color = m_tonemapping->ldr_output();
    }

    return {display_color, hdr.color, hdr.depth};
}

float& IRenderer::exposure() {
    INVARIANT(m_tonemapping);
    return m_tonemapping->m_exposure;
}

uint32_t& IRenderer::tone_map_mode() {
    INVARIANT(m_tonemapping);
    return m_tonemapping->m_mode;
}

bool& IRenderer::auto_exposure() {
    INVARIANT(m_tonemapping);
    return m_tonemapping->m_auto_exposure;
}

float& IRenderer::adaptation_speed() {
    INVARIANT(m_tonemapping);
    return m_tonemapping->m_adaptation_speed;
}

void IRenderer::draw_imgui() {
    if (!ImGui::CollapsingHeader(name().data(), ImGuiTreeNodeFlags_DefaultOpen)) return;
    for (auto& c : m_children) {
        if (ImGui::TreeNodeEx(c->name().data(), ImGuiTreeNodeFlags_DefaultOpen)) {
            c->draw_imgui();
            ImGui::TreePop();
        }
    }
    do_draw_imgui();
    if (m_tonemapping && ImGui::TreeNodeEx("Tone Mapping", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enabled", &m_tonemapping_enabled);
        if (m_tonemapping_enabled) m_tonemapping->draw_imgui();
        ImGui::TreePop();
    }
}

}  // namespace pts::rendering
