#pragma once

#include <core/rendering/scenePass.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <string_view>
#include <variant>
#include <vector>

namespace pts::editor {

/// GPU-aligned flattened triangle for brute-force ray intersection.
struct PackedTriangle {
    glm::vec3 v0;
    uint32_t _pad0{};
    glm::vec3 v1;
    uint32_t _pad1{};
    glm::vec3 v2;
    uint32_t _pad2{};
    glm::vec3 n0;
    uint32_t _pad3{};
    glm::vec3 n1;
    uint32_t _pad4{};
    glm::vec3 n2;
    uint32_t material_index{UINT32_MAX};
    uint32_t _pad5[4]{};
};
static_assert(sizeof(PackedTriangle) == 112, "PackedTriangle must be 112 bytes for GPU alignment");

class PathTracerPass final : public rendering::IScenePass {
   public:
    using IScenePass::IScenePass;
    ~PathTracerPass() override;

    PathTracerPass(const PathTracerPass&) = delete;
    PathTracerPass& operator=(const PathTracerPass&) = delete;
    PathTracerPass(PathTracerPass&&) = delete;
    PathTracerPass& operator=(PathTracerPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto is_ready() const noexcept -> bool override;

    void do_setup(const webgpu::Device& device) override;
    void add_to_frame_graph(rendering::FrameGraph& fg, const rendering::PassContext& ctx) override;
    void draw_viewport_controls() override;

   private:
    void rebuild_scene_buffer(const webgpu::Device& device, WGPUQueue queue,
                              const rendering::RenderWorld& world);
    void ensure_pixel_buffers(const webgpu::Device& device, uint32_t width, uint32_t height);

    struct Ready {
        webgpu::ShaderModule compute_shader;
        webgpu::ComputePipeline compute_pipeline;
        webgpu::Buffer uniform_buffer;
        WGPUBindGroupLayout compute_bgl = nullptr;

        webgpu::ShaderModule blit_shader;
        webgpu::RenderPipeline blit_pipeline;
        webgpu::Buffer blit_uniform_buffer;
        WGPUBindGroupLayout blit_bgl = nullptr;
    };

    std::variant<std::monostate, Ready> m_state;

    // Scene data — flattened triangles built from RenderWorld mesh data
    std::vector<PackedTriangle> m_scene_triangles;
    webgpu::Buffer m_scene_buffer;
    uint32_t m_cached_mesh_version = UINT32_MAX;
    uint32_t m_scene_triangle_count = 0;

    // Per-pixel buffers
    webgpu::Buffer m_accum_buffer;
    webgpu::Buffer m_output_buffer;
    uint32_t m_pixel_width = 0;
    uint32_t m_pixel_height = 0;

    // Camera change detection
    glm::mat4 m_prev_vp{0.0f};
    uint32_t m_frame_count = 0;
};

}  // namespace pts::editor
