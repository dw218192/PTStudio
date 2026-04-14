#include "lobePass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/webgpu/device.h>
#include <imgui.h>
#include <lobe_shader_metadata.h>

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace pts;
using namespace pts::editor;

struct LobeUniforms {
    glm::mat4 mvp;
    glm::vec3 light_dir;
    float roughness;
    float metallic;
    float scale;
    uint32_t grid_cols;
    uint32_t grid_rows;
    uint32_t mode;
    uint32_t _pad[3];
};
static_assert(sizeof(LobeUniforms) == 112, "LobeUniforms must match shader std140 layout");
static_assert(LobePass::k_uniform_align >= sizeof(LobeUniforms),
              "Alignment must be >= uniform struct size");

auto LobePass::name() const noexcept -> std::string_view {
    return "lobe";
}

void LobePass::render(rendering::FrameGraph& fg, const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;
    ensure_initialized(ctx.device);

    auto descriptor_layout = fg.bind_group_layout(
        "lobe/desc", editor_lobe_shader::create_bind_group_layout_0(ctx.device.handle()));

    auto* pipeline_handle = fg.render_pipeline("lobe")
                                .shader("editor/generated/shaders/lobe.wgsl")
                                .color_format(WGPUTextureFormat_RGBA8Unorm)
                                .depth_format(WGPUTextureFormat_Depth32Float)
                                .depth_write(true)
                                .depth_compare(WGPUCompareFunction_Less)
                                .cull_mode(WGPUCullMode_None)
                                .bind_group_layouts({descriptor_layout})
                                .build();

    // Register uniform buffer (2 aligned slots: specular + diffuse)
    rendering::BufferDesc buf_desc{};
    buf_desc.size = k_uniform_align * 2;
    buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_decl = create_buffer(fg, buf_desc, "uniforms");

    // Register descriptor
    auto bg_decl = descriptor(fg, descriptor_layout, "bg0")
                       .buffer(0, uniform_buf_decl, 0, sizeof(LobeUniforms))
                       .build();

    rendering::TextureDesc color_desc;
    color_desc.width = k_texture_size;
    color_desc.height = k_texture_size;
    color_desc.format = WGPUTextureFormat_RGBA8Unorm;
    color_desc.clear_color = {0.1, 0.1, 0.1, 1.0};

    rendering::TextureDesc depth_desc;
    depth_desc.width = k_texture_size;
    depth_desc.height = k_texture_size;
    depth_desc.format = WGPUTextureFormat_Depth32Float;

    auto color_decl = fg.texture("lobe_color", color_desc);
    auto depth_decl = create_texture(fg, depth_desc, "depth");
    m_lobe_color_decl = color_decl;

    // Fixed camera looking at origin
    auto eye = glm::vec3(0.0f, -2.5f, 1.5f);
    auto center = glm::vec3(0.0f);
    auto up = glm::vec3(0.0f, 0.0f, 1.0f);
    auto view_mat = glm::lookAt(eye, center, up);
    auto proj_mat = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
    auto mvp = proj_mat * view_mat;

    // Compute light direction from azimuth/elevation
    float az = glm::radians(m_light_azimuth_deg);
    float el = glm::radians(m_light_elevation_deg);
    auto light_dir =
        glm::vec3(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));

    auto queue = ctx.queue;
    auto roughness = m_roughness;
    auto metallic = m_metallic;
    auto scale = m_scale;
    auto show_specular = m_show_specular;
    auto show_diffuse = m_show_diffuse;

    fg.add_pass("lobe")
        .color(color_decl)
        .depth(depth_decl)
        .execute([=](rendering::ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto uniform_buf = exec.get(uniform_buf_decl).buffer;
            auto desc_group = exec.get(bg_decl).bind_group;

            // Upload both uniform slots
            LobeUniforms lu_spec{};
            lu_spec.mvp = mvp;
            lu_spec.light_dir = light_dir;
            lu_spec.roughness = roughness;
            lu_spec.metallic = metallic;
            lu_spec.scale = scale;
            lu_spec.grid_cols = k_grid_cols;
            lu_spec.grid_rows = k_grid_rows;
            lu_spec.mode = 0;

            LobeUniforms lu_diff = lu_spec;
            lu_diff.mode = 1;

            wgpuQueueWriteBuffer(queue, uniform_buf, 0, &lu_spec, sizeof(lu_spec));
            wgpuQueueWriteBuffer(queue, uniform_buf, k_uniform_align, &lu_diff, sizeof(lu_diff));

            uint32_t vertex_count = (k_grid_cols - 1) * (k_grid_rows - 1) * 6;
            wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);

            if (show_specular) {
                uint32_t offset_spec = 0;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, desc_group, 1, &offset_spec);
                wgpuRenderPassEncoderDraw(pass, vertex_count, 1, 0, 0);
            }

            if (show_diffuse) {
                uint32_t offset_diff = k_uniform_align;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, desc_group, 1, &offset_diff);
                wgpuRenderPassEncoderDraw(pass, vertex_count, 1, 0, 0);
            }
        });
}

void LobePass::update_texture_refs(rendering::FrameGraph& fg) {
    if (m_lobe_color_decl) {
        auto* compiled = fg.compiled_texture(m_lobe_color_decl);
        if (compiled) {
            m_lobe_color_view = compiled->view;
        }
    }
}

/// Project a 3D point to 2D image-space coordinates within the lobe texture.
/// Returns the 2D position relative to the image top-left, or nullopt if behind camera.
static std::optional<ImVec2> project_to_image(const glm::vec3& world_pos, const glm::mat4& mvp,
                                              const ImVec2& img_min, float img_size) {
    auto clip = mvp * glm::vec4(world_pos, 1.0f);
    if (clip.w <= 0.0f) return std::nullopt;
    auto ndc = glm::vec3(clip) / clip.w;
    float sx = img_min.x + (ndc.x * 0.5f + 0.5f) * img_size;
    float sy = img_min.y + (1.0f - (ndc.y * 0.5f + 0.5f)) * img_size;
    return ImVec2(sx, sy);
}

void LobePass::set_material(float roughness, float metallic) {
    m_roughness = roughness;
    m_metallic = metallic;
}

void LobePass::draw_imgui() {
    // No standalone window -- lobe is drawn inline via draw_lobe_widget()
}

bool LobePass::draw_lobe_widget() {
    bool changed = false;
    changed |= ImGui::SliderFloat("Roughness", &m_roughness, 0.01f, 1.0f);
    changed |= ImGui::SliderFloat("Metallic", &m_metallic, 0.0f, 1.0f);
    ImGui::SliderFloat("Scale", &m_scale, 0.1f, 5.0f);
    ImGui::SliderFloat("Light Azimuth", &m_light_azimuth_deg, -180.0f, 180.0f);
    ImGui::SliderFloat("Light Elevation", &m_light_elevation_deg, 0.0f, 90.0f);
    ImGui::Checkbox("Show Specular", &m_show_specular);
    ImGui::SameLine();
    ImGui::Checkbox("Show Diffuse", &m_show_diffuse);

    if (!m_lobe_color_view) return changed;

    auto img_pos = ImGui::GetCursorScreenPos();
    float avail = ImGui::GetContentRegionAvail().x;
    float img_size = std::min(avail, static_cast<float>(k_texture_size));
    ImGui::Image(reinterpret_cast<ImTextureID>(m_lobe_color_view), ImVec2(img_size, img_size));

    // Draw light direction arrow overlaid on the image
    auto eye = glm::vec3(0.0f, -2.5f, 1.5f);
    auto view_mat = glm::lookAt(eye, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    auto proj_mat = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
    auto mvp = proj_mat * view_mat;

    float az = glm::radians(m_light_azimuth_deg);
    float el = glm::radians(m_light_elevation_deg);
    auto light_dir =
        glm::vec3(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));

    float arrow_len = 1.2f;
    auto tip = light_dir * arrow_len;
    auto base = light_dir * 0.3f;

    auto p_tip = project_to_image(tip, mvp, img_pos, img_size);
    auto p_base = project_to_image(base, mvp, img_pos, img_size);

    if (p_tip && p_base) {
        auto* draw_list = ImGui::GetWindowDrawList();
        auto yellow = IM_COL32(255, 220, 50, 255);
        draw_list->AddLine(*p_base, *p_tip, yellow, 2.0f);

        auto dir = ImVec2(p_tip->x - p_base->x, p_tip->y - p_base->y);
        float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
        if (len > 1.0f) {
            dir.x /= len;
            dir.y /= len;
            float head_size = 8.0f;
            ImVec2 perp(-dir.y, dir.x);
            ImVec2 a(p_tip->x - dir.x * head_size + perp.x * head_size * 0.4f,
                     p_tip->y - dir.y * head_size + perp.y * head_size * 0.4f);
            ImVec2 b(p_tip->x - dir.x * head_size - perp.x * head_size * 0.4f,
                     p_tip->y - dir.y * head_size - perp.y * head_size * 0.4f);
            draw_list->AddTriangleFilled(*p_tip, a, b, yellow);
        }
    }
    return changed;
}
