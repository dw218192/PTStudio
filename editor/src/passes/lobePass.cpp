#include "lobePass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <imgui.h>

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

LobePass::~LobePass() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group) {
            wgpuBindGroupRelease(ready->bind_group);
        }
        if (ready->bind_group_layout) {
            wgpuBindGroupLayoutRelease(ready->bind_group_layout);
        }
    }
}

auto LobePass::name() const noexcept -> std::string_view {
    return "lobe";
}

auto LobePass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

void LobePass::setup(const webgpu::Device& device) {
    PRECONDITION_MSG(m_shader_loader, "shader loader not set");

    // Release existing state for re-entry (hot-reload)
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group) wgpuBindGroupRelease(ready->bind_group);
        if (ready->bind_group_layout) wgpuBindGroupLayoutRelease(ready->bind_group_layout);
    }

    auto shader_src = m_shader_loader->load("editor/generated/shaders/lobe.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    // Uniform buffer holds 2 aligned copies (specular + diffuse)
    auto uniform_buffer = device.create_buffer(
        k_uniform_align * 2,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    // Create bind group layout with dynamic offset for dual draw
    WGPUBindGroupLayoutEntry bgl_entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    bgl_entry.binding = 0;
    bgl_entry.visibility =
        static_cast<WGPUShaderStage>(WGPUShaderStage_Vertex | WGPUShaderStage_Fragment);
    bgl_entry.buffer.type = WGPUBufferBindingType_Uniform;
    bgl_entry.buffer.hasDynamicOffset = true;
    bgl_entry.buffer.minBindingSize = sizeof(LobeUniforms);

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = 1;
    bgl_desc.entries = &bgl_entry;
    auto bind_group_layout = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);

    WGPUBindGroupEntry bg_entry = WGPU_BIND_GROUP_ENTRY_INIT;
    bg_entry.binding = 0;
    bg_entry.buffer = uniform_buffer.handle();
    bg_entry.offset = 0;
    bg_entry.size = sizeof(LobeUniforms);

    WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg_desc.layout = bind_group_layout;
    bg_desc.entryCount = 1;
    bg_desc.entries = &bg_entry;
    auto bind_group = wgpuDeviceCreateBindGroup(device.handle(), &bg_desc);

    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &bind_group_layout;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

    auto pipeline = webgpu::RenderPipelineBuilder(device)
                        .shader(shader)
                        .color_format(WGPUTextureFormat_RGBA8Unorm)
                        .depth_format(WGPUTextureFormat_Depth24Plus)
                        .depth_write(true)
                        .depth_compare(WGPUCompareFunction_Less)
                        .cull_mode(WGPUCullMode_None)
                        .pipeline_layout(pipeline_layout)
                        .build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    m_state = Ready{
        std::move(shader), std::move(pipeline), std::move(uniform_buffer),
        bind_group,        bind_group_layout,
    };
}

void LobePass::add_to_frame_graph(rendering::FrameGraph& fg, const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    rendering::TextureDesc color_desc;
    color_desc.width = k_texture_size;
    color_desc.height = k_texture_size;
    color_desc.format = WGPUTextureFormat_RGBA8Unorm;
    color_desc.clear_color = {0.1, 0.1, 0.1, 1.0};

    rendering::TextureDesc depth_desc;
    depth_desc.width = k_texture_size;
    depth_desc.height = k_texture_size;
    depth_desc.format = WGPUTextureFormat_Depth24Plus;

    auto color = fg.find_or_create("lobe_color", color_desc);
    auto depth = fg.find_or_create("lobe_depth", depth_desc);
    m_lobe_color_handle = color;

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
    auto* pipeline_handle = ready.pipeline.handle();
    auto uniform_buf = ready.uniform_buffer.handle();
    auto bind_group = ready.bind_group;
    auto roughness = m_roughness;
    auto metallic = m_metallic;
    auto scale = m_scale;
    auto show_specular = m_show_specular;
    auto show_diffuse = m_show_diffuse;

    // Upload both uniform slots before the render pass
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

    fg.add_pass("lobe").color(color).depth(depth).execute([=](WGPURenderPassEncoder pass) {
        uint32_t vertex_count = (k_grid_cols - 1) * (k_grid_rows - 1) * 6;
        wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);

        if (show_specular) {
            uint32_t offset_spec = 0;
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 1, &offset_spec);
            wgpuRenderPassEncoderDraw(pass, vertex_count, 1, 0, 0);
        }

        if (show_diffuse) {
            uint32_t offset_diff = k_uniform_align;
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 1, &offset_diff);
            wgpuRenderPassEncoderDraw(pass, vertex_count, 1, 0, 0);
        }
    });
}

void LobePass::update_texture_refs(rendering::FrameGraph& fg) {
    if (m_lobe_color_handle.is_valid()) {
        m_lobe_color_ref = fg.get_texture_ref(m_lobe_color_handle);
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

void LobePass::draw_imgui() {
    if (ImGui::Begin("BRDF Lobe")) {
        ImGui::SliderFloat("Roughness", &m_roughness, 0.01f, 1.0f);
        ImGui::SliderFloat("Metallic", &m_metallic, 0.0f, 1.0f);
        ImGui::SliderFloat("Scale", &m_scale, 0.1f, 5.0f);
        ImGui::SliderFloat("Light Azimuth", &m_light_azimuth_deg, -180.0f, 180.0f);
        ImGui::SliderFloat("Light Elevation", &m_light_elevation_deg, 0.0f, 90.0f);
        ImGui::Checkbox("Show Specular", &m_show_specular);
        ImGui::Checkbox("Show Diffuse", &m_show_diffuse);
        ImGui::Separator();
        if (m_lobe_color_ref) {
            auto img_pos = ImGui::GetCursorScreenPos();
            float img_size = static_cast<float>(k_texture_size);
            ImGui::Image(reinterpret_cast<ImTextureID>(m_lobe_color_ref.view()),
                         ImVec2(img_size, img_size));

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

                // Arrowhead
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
        }
    }
    ImGui::End();
}
