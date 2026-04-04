#pragma once

#include <core/rendering/renderPass.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/device.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <algorithm>
#include <boost/core/span.hpp>
#include <variant>
#include <vector>

namespace pts::rendering {

// ── ShadowPassData ──────────────────────────────────────────────────────────
// Cross-pass output of ShadowMapPass. Written here, read by renderers via
// RenderWorld::pass_data_for(&ShadowPassData::k_key).

struct ShadowPassData {
    static inline const char k_key = 0;

    webgpu::Buffer info_buffer;
    uint32_t count = 0;

    void upload(boost::span<const ShadowInfo> infos, const webgpu::Device& device,
                WGPUQueue queue) {
        constexpr std::size_t k_min_size = sizeof(ShadowInfo);
        auto info_bytes =
            std::max(k_min_size, static_cast<std::size_t>(infos.size()) * sizeof(ShadowInfo));
        if (!info_buffer.is_valid() || info_buffer.size() < info_bytes) {
            info_buffer = device.create_buffer(
                info_bytes,
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
        }
        if (!infos.empty()) {
            wgpuQueueWriteBuffer(queue, info_buffer.handle(), 0, infos.data(),
                                 infos.size() * sizeof(ShadowInfo));
        }
        uint32_t active = 0;
        for (const auto& si : infos) {
            if (si.has_shadow) ++active;
        }
        count = active;
    }

    static ShadowPassData& get_or_create(RenderWorld& world) {
        auto& map = world.pass_data_for(&k_key);
        auto it = map.find(0);
        if (it == map.end()) {
            it = map.emplace(0, RenderWorld::PassDataEntry{}).first;
        }
        auto& entry = it->second;
        if (!entry.data) {
            entry.data = RenderWorld::ErasedPtr(
                new ShadowPassData{}, [](void* p) { delete static_cast<ShadowPassData*>(p); });
        }
        return *static_cast<ShadowPassData*>(entry.data.get());
    }

    static ShadowPassData* find(RenderWorld& world) {
        auto& map = world.pass_data_for(&k_key);
        auto it = map.find(0);
        if (it == map.end() || !it->second.data) return nullptr;
        return static_cast<ShadowPassData*>(it->second.data.get());
    }
};

// ── ShadowMapPass ───────────────────────────────────────────────────────────

inline constexpr uint32_t k_max_shadow_maps = 4;
inline constexpr uint32_t k_default_shadow_resolution = 2048;

/// Renders depth maps for shadow-casting distant lights.
class ShadowMapPass final : public IRenderPass {
   public:
    explicit ShadowMapPass(const ShaderLoader& sl);
    ~ShadowMapPass() override;

    ShadowMapPass(const ShadowMapPass&) = delete;
    ShadowMapPass& operator=(const ShadowMapPass&) = delete;
    ShadowMapPass(ShadowMapPass&&) = delete;
    ShadowMapPass& operator=(ShadowMapPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "shadow_map";
    }
    [[nodiscard]] auto is_ready() const noexcept -> bool override;
    [[nodiscard]] auto requires_viewport() const noexcept -> bool override {
        return false;
    }

    void do_setup(const webgpu::Device& device) override;
    void add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) override;
    void draw_imgui() override;

    [[nodiscard]] WGPUTextureView shadow_array_view() const;
    [[nodiscard]] bool enabled() const {
        return m_enabled;
    }

   private:
    bool m_enabled = true;
    static constexpr uint32_t k_uniform_align = 256;

    struct Ready {
        webgpu::ShaderModule shader;
        webgpu::RenderPipeline pipeline;
        webgpu::Buffer per_object_uniform_buf;
        WGPUBindGroupLayout bgl = nullptr;
        WGPUBindGroup bind_group = nullptr;
        uint32_t object_capacity = 0;
    };
    std::variant<std::monostate, Ready> m_state;

    // Shadow texture array (managed by this pass, not FrameGraph)
    WGPUTexture m_shadow_texture = nullptr;
    WGPUTextureView m_shadow_array_view = nullptr;      // full array view for sampling
    std::vector<WGPUTextureView> m_shadow_layer_views;  // per-layer views for rendering
    uint32_t m_current_layer_count = 0;

    uint32_t m_resolution = k_default_shadow_resolution;

    void ensure_shadow_texture(const webgpu::Device& device, uint32_t layer_count);
    void release_shadow_texture();
};

}  // namespace pts::rendering
