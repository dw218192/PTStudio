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

void LobePass::do_setup(const webgpu::Device& device) {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        if (ready->bind_group_layout) wgpuBindGroupLayoutRelease(ready->bind_group_layout);
    }

    auto shader_src = get_shader_loader().load("editor/generated/shaders/lobe.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

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

    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &bind_group_layout;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

    auto pipeline = webgpu::RenderPipelineBuilder(device)
                        .shader(shader)
                        .color_format(WGPUTextureFormat_RGBA8Unorm)
                        .depth_format(WGPUTextureFormat_Depth32Float)
                        .depth_write(true)
                        .depth_compare(WGPUCompareFunction_Less)
                        .cull_mode(WGPUCullMode_None)
                        .pipeline_layout(pipeline_layout)
                        .build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    m_state = Ready{
        std::move(shader),
        std::move(pipeline),
        bind_group_layout,
    };
}

void LobePass::render(rendering::FrameGraph& fg, const rendering::PassContext& ctx) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);

    // Register uniform buffer (2 aligned slots: specular + diffuse)
    rendering::BufferDesc buf_desc{};
    buf_desc.size = k_uniform_align * 2;
    buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_handle = create_buffer(fg, buf_desc, "uniforms");

    // Register bind group
    rendering::BindGroupDesc bg_desc{};
    bg_desc.layout = ready.bind_group_layout;
    bg_desc.entries = {
        {0, rendering::ManagedBufferBinding{uniform_buf_handle, 0, sizeof(LobeUniforms)}}};
    auto bg_handle = create_bind_group(fg, std::move(bg_desc), "bg0");

    rendering::TextureDesc color_desc;
    color_desc.width = k_texture_size;
    color_desc.height = k_texture_size;
    color_desc.format = WGPUTextureFormat_RGBA8Unorm;
    color_desc.clear_color = {0.1, 0.1, 0.1, 1.0};

    rendering::TextureDesc depth_desc;
    depth_desc.width = k_texture_size;
    depth_desc.height = k_texture_size;
    depth_desc.format = WGPUTextureFormat_Depth32Float;

    auto color = fg.find_or_create("lobe_color", color_desc);
    auto depth = create_texture(fg, depth_desc, "depth");
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
    auto roughness = m_roughness;
    auto metallic = m_metallic;
    auto scale = m_scale;
    auto show_specular = m_show_specular;
    auto show_diffuse = m_show_diffuse;

    fg.add_pass("lobe").color(color).depth(depth).execute([=, &fg](WGPURenderPassEncoder pass) {
        auto uniform_buf = fg.get_buffer_ref(uniform_buf_handle).handle();
        auto bind_group = fg.get_bind_group_ref(bg_handle).handle();

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

void LobePass::set_material(float roughness, float metallic) {
    m_roughness = roughness;
    m_metallic = metallic;
}

void LobePass::draw_imgui() {
    // No standalone window — lobe is drawn inline via draw_lobe_widget()
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

    if (!m_lobe_color_ref) return changed;

    auto img_pos = ImGui::GetCursorScreenPos();
    float avail = ImGui::GetContentRegionAvail().x;
    float img_size = std::min(avail, static_cast<float>(k_texture_size));
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
