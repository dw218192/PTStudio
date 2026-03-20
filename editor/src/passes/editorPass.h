#pragma once

#include <core/rendering/scenePass.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <string_view>
#include <variant>

namespace pts::editor {

/// Combined picking + wireframe light gizmo pass.
/// Submits two frame graph passes:
///   "editor_picking" — renders mesh objects + light shapes to picking_ids
///   "editor_gizmos"  — renders light wireframe shapes to scene_color
class EditorPass final : public rendering::IScenePass {
   public:
    using IScenePass::IScenePass;
    ~EditorPass() override;

    EditorPass(const EditorPass&) = delete;
    EditorPass& operator=(const EditorPass&) = delete;
    EditorPass(EditorPass&&) = delete;
    EditorPass& operator=(EditorPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto is_ready() const noexcept -> bool override;

    void do_setup(const webgpu::Device& device) override;
    void add_to_frame_graph(rendering::FrameGraph& fg, const rendering::PassContext& ctx) override;

    static constexpr uint32_t k_uniform_align = 256;

   private:
    void ensure_picking_capacity(const webgpu::Device& device, uint32_t count);
    void ensure_gizmo_capacity(const webgpu::Device& device, uint32_t count);

    struct GizmoMesh {
        webgpu::Buffer vertex_buffer;
        uint32_t vertex_count = 0;
    };

    struct Ready {
        // Mesh picking pipeline (reuses picking shader)
        webgpu::ShaderModule picking_shader;
        webgpu::RenderPipeline picking_pipeline;
        webgpu::Buffer picking_uniform_buffer;
        WGPUBindGroup picking_bind_group = nullptr;
        WGPUBindGroupLayout picking_bind_group_layout = nullptr;
        uint32_t picking_capacity = 0;

        // Gizmo pipelines (color + picking for light shapes)
        webgpu::ShaderModule gizmo_shader;
        webgpu::RenderPipeline gizmo_color_pipeline;    // scene_color, LineList, blend
        webgpu::RenderPipeline gizmo_picking_pipeline;  // picking_ids, LineList
        webgpu::Buffer gizmo_uniform_buffer;
        WGPUBindGroup gizmo_bind_group = nullptr;
        WGPUBindGroupLayout gizmo_bind_group_layout = nullptr;
        uint32_t gizmo_capacity = 0;
    };

    std::variant<std::monostate, Ready> m_state;
};

}  // namespace pts::editor
