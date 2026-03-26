#pragma once

#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>
#include <pxr/usd/sdf/path.h>

#include <algorithm>
#include <cstdint>
#include <variant>
#include <vector>

namespace pts::editor {

/// Compute a uniform scale factor so a gizmo with the given world-space
/// radius maintains a minimum screen-space size at the given camera distance.
/// The factor is clamped to >= 1 (never shrinks the gizmo).
inline float gizmo_distance_scale(float camera_distance, float world_radius,
                                  float min_screen_radius = 0.05f) {
    float r = std::max(world_radius, 0.1f);
    return std::max(1.0f, min_screen_radius * camera_distance / r);
}

/// Combined picking + wireframe light gizmo pass.
/// Submits two frame graph passes:
///   "editor_picking" — renders mesh objects + light shapes to picking_ids
///   "editor_gizmos"  — renders light wireframe shapes to scene_color
class EditorPass final : public rendering::IRenderPass {
   public:
    using IRenderPass::IRenderPass;
    ~EditorPass() override;

    EditorPass(const EditorPass&) = delete;
    EditorPass& operator=(const EditorPass&) = delete;
    EditorPass(EditorPass&&) = delete;
    EditorPass& operator=(EditorPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto is_ready() const noexcept -> bool override;

    void do_setup(const webgpu::Device& device) override;
    void add_to_frame_graph(rendering::FrameGraph& fg, const rendering::PassContext& ctx) override;

    /// Resolve a picking ID to its prim path. Returns empty path if invalid.
    /// Valid after add_to_frame_graph has run for the current frame.
    [[nodiscard]] auto resolve_picking_id(uint32_t id) const noexcept -> const pxr::SdfPath&;

    /// Find the picking ID for a prim path. Returns UINT32_MAX if not found.
    [[nodiscard]] auto find_picking_id(const pxr::SdfPath& prim_path) const noexcept -> uint32_t;

    static constexpr uint32_t k_uniform_align = 256;

   private:
    void ensure_picking_capacity(const webgpu::Device& device, uint32_t count);
    void ensure_gizmo_capacity(const webgpu::Device& device, uint32_t count);

    struct GizmoMesh {
        webgpu::Buffer vertex_buffer;  // line-list for color overlay
        uint32_t vertex_count = 0;
        webgpu::Buffer pick_vertex_buffer;  // triangle-list for picking
        uint32_t pick_vertex_count = 0;
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

    /// Flat table: picking_id → prim_path. Built each frame in add_to_frame_graph.
    std::vector<pxr::SdfPath> m_picking_table;
};

}  // namespace pts::editor
