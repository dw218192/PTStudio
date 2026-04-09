#pragma once

#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/webgpu.h>
#include <pxr/usd/sdf/path.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/gtc/constants.hpp>
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

// ── Gizmo geometry generation (inline for testability) ────────────────

static constexpr uint32_t k_gizmo_circle_segments = 48;

inline void generate_gizmo_circle(std::vector<glm::vec3>& out, glm::vec3 center, glm::vec3 axis_a,
                                  glm::vec3 axis_b, float radius) {
    for (uint32_t i = 0; i < k_gizmo_circle_segments; ++i) {
        float a0 = glm::two_pi<float>() * static_cast<float>(i) / k_gizmo_circle_segments;
        float a1 = glm::two_pi<float>() * static_cast<float>(i + 1) / k_gizmo_circle_segments;
        out.push_back(center + (std::cos(a0) * axis_a + std::sin(a0) * axis_b) * radius);
        out.push_back(center + (std::cos(a1) * axis_a + std::sin(a1) * axis_b) * radius);
    }
}

/// Generate line-list wireframe vertices for a light gizmo.
/// Returns an empty vector for light types that have no gizmo (e.g. Dome).
inline std::vector<glm::vec3> generate_light_verts(const rendering::LightData& light) {
    std::vector<glm::vec3> verts;
    switch (light.type) {
        case rendering::LightData::Type::Sphere: {
            float r = std::max(light.radius, 0.1f);
            verts.reserve(k_gizmo_circle_segments * 2 * 3);
            generate_gizmo_circle(verts, {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, r);
            generate_gizmo_circle(verts, {0, 0, 0}, {1, 0, 0}, {0, 0, 1}, r);
            generate_gizmo_circle(verts, {0, 0, 0}, {0, 1, 0}, {0, 0, 1}, r);
            break;
        }
        case rendering::LightData::Type::Rect: {
            float hw = light.width * 0.5f;
            float hh = light.height * 0.5f;
            float arrow = std::min(hw, hh) * 0.7f;
            verts = {{-hw, -hh, 0},
                     {hw, -hh, 0},
                     {hw, -hh, 0},
                     {hw, hh, 0},
                     {hw, hh, 0},
                     {-hw, hh, 0},
                     {-hw, hh, 0},
                     {-hw, -hh, 0},
                     // Direction arrow along -Z (emission direction)
                     {0, 0, 0},
                     {0, 0, -arrow}};
            break;
        }
        case rendering::LightData::Type::Disk: {
            float r = std::max(light.radius, 0.1f);
            float arrow = r * 0.7f;
            verts.reserve(k_gizmo_circle_segments * 2 + 2);
            generate_gizmo_circle(verts, {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, r);
            // Direction arrow along -Z (emission direction)
            verts.push_back({0, 0, 0});
            verts.push_back({0, 0, -arrow});
            break;
        }
        case rendering::LightData::Type::Distant: {
            constexpr float r = 0.5f;
            constexpr float arrow_len = 1.0f;
            constexpr float head_len = 0.2f;
            constexpr float head_r = 0.1f;
            verts.reserve(k_gizmo_circle_segments * 2 + 10);
            generate_gizmo_circle(verts, {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, r);
            // Direction arrow along -Z (light direction in local space)
            verts.push_back({0, 0, 0});
            verts.push_back({0, 0, -arrow_len});
            // Arrowhead
            verts.push_back({0, 0, -arrow_len});
            verts.push_back({head_r, 0, -arrow_len + head_len});
            verts.push_back({0, 0, -arrow_len});
            verts.push_back({-head_r, 0, -arrow_len + head_len});
            verts.push_back({0, 0, -arrow_len});
            verts.push_back({0, head_r, -arrow_len + head_len});
            verts.push_back({0, 0, -arrow_len});
            verts.push_back({0, -head_r, -arrow_len + head_len});
            break;
        }
        case rendering::LightData::Type::Dome:
            break;
    }
    return verts;
}

/// Combined picking + wireframe light gizmo pass.
/// Submits two frame graph passes:
///   "editor_picking" — renders mesh objects + light shapes to picking_ids
///   "editor_gizmos"  — renders light wireframe shapes to scene_color
class EditorPass final : public rendering::IPass {
   public:
    using IPass::IPass;
    ~EditorPass() override;

    EditorPass(const EditorPass&) = delete;
    EditorPass& operator=(const EditorPass&) = delete;
    EditorPass(EditorPass&&) = delete;
    EditorPass& operator=(EditorPass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto is_ready() const noexcept -> bool override;

    void do_setup(const webgpu::Device& device) override;
    void render(rendering::FrameGraph& fg, const rendering::PassContext& ctx);

    /// Resolve a picking ID to its prim path. Returns empty path if invalid.
    /// Valid after add_to_frame_graph has run for the current frame.
    [[nodiscard]] auto resolve_picking_id(uint32_t id) const noexcept -> const pxr::SdfPath&;

    /// Find the picking ID for a prim path. Returns UINT32_MAX if not found.
    [[nodiscard]] auto find_picking_id(const pxr::SdfPath& prim_path) const noexcept -> uint32_t;

    static constexpr uint32_t k_uniform_align = 256;

   private:
    struct GizmoMesh {
        webgpu::Buffer vertex_buffer;  // line-list for color overlay
        uint32_t vertex_count = 0;
    };

    struct Ready {
        // Mesh picking pipeline (reuses picking shader)
        webgpu::ShaderModule picking_shader;
        webgpu::RenderPipeline picking_pipeline;
        webgpu::RenderPipeline picking_line_pipeline;  // LineList topology for wireframe picking
        WGPUBindGroupLayout picking_descriptor_layout = nullptr;

        // Gizmo pipeline (wireframe color overlay for light shapes)
        webgpu::ShaderModule gizmo_shader;
        webgpu::RenderPipeline gizmo_color_pipeline;  // scene_color, LineList, blend
        WGPUBindGroupLayout gizmo_descriptor_layout = nullptr;
    };

    std::variant<std::monostate, Ready> m_state;

    /// Flat table: picking_id → prim_path. Built each frame in add_to_frame_graph.
    std::vector<pxr::SdfPath> m_picking_table;
};

}  // namespace pts::editor
