#pragma once

#include <core/rendering/ltcTextures.h>
#include <core/rendering/renderer.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <string_view>
#include <variant>

namespace pts::editor {

class ForwardPass final : public rendering::IRenderer {
   public:
    explicit ForwardPass(const rendering::ShaderLoader& sl);
    ~ForwardPass() override;

    ForwardPass(const ForwardPass&) = delete;
    ForwardPass& operator=(const ForwardPass&) = delete;
    ForwardPass(ForwardPass&&) = delete;
    ForwardPass& operator=(ForwardPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto is_ready() const noexcept -> bool override;
    [[nodiscard]] auto debug_target_names() const noexcept
        -> std::pair<const char* const*, uint32_t> override;

    void do_renderer_setup(const webgpu::Device& device) override;
    void do_add_to_frame_graph(rendering::FrameGraph& fg,
                               const rendering::PassContext& ctx) override;

    static constexpr uint32_t k_uniform_align = 256;

   private:
    bool ensure_capacity(const webgpu::Device& device, uint32_t object_count);

    struct Ready {
        webgpu::ShaderModule shader;
        webgpu::RenderPipeline pipeline;
        webgpu::Buffer uniform_buffer;
        WGPUBindGroup bind_group = nullptr;
        WGPUBindGroupLayout bind_group_layout = nullptr;
        uint32_t capacity = 0;
        WGPUBuffer cached_light_buf = nullptr;
        WGPUBuffer cached_material_buf = nullptr;
        WGPUTextureView cached_texture_view = nullptr;
        WGPUSampler cached_texture_sampler = nullptr;
        rendering::LtcTextures ltc_textures;
        // Shadow receiver resources (bind group 1)
        WGPUBindGroupLayout shadow_recv_bgl = nullptr;
        WGPUSampler shadow_sampler = nullptr;
    };

    std::variant<std::monostate, Ready> m_state;
};

}  // namespace pts::editor
