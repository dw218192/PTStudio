#include "lobePass.h"

#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <imgui.h>
#include <lobe_shader_metadata.h>

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "editorResources.h"

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
};
static_assert(sizeof(LobeUniforms) == 96, "LobeUniforms must match shader std140 layout");

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
    auto shader_src = editor_resources::get_resource("editor/generated/shaders/lobe.wgsl");
    PRECONDITION_MSG(shader_src, "Missing embedded resource: editor/generated/shaders/lobe.wgsl");

    auto shader = device.create_shader_module_from_source(*shader_src);

    auto uniform_buffer = device.create_buffer(
        sizeof(LobeUniforms),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));

    auto bind_group_layout = editor_lobe_shader::create_bind_group_layout_0(device.handle());

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

    fg.add_pass("lobe").color(color).depth(depth).execute([=](WGPURenderPassEncoder pass) {
        LobeUniforms lu;
        lu.mvp = mvp;
        lu.light_dir = light_dir;
        lu.roughness = roughness;
        lu.metallic = metallic;
        lu.scale = scale;
        lu.grid_cols = k_grid_cols;
        lu.grid_rows = k_grid_rows;
        wgpuQueueWriteBuffer(queue, uniform_buf, 0, &lu, sizeof(lu));

        wgpuRenderPassEncoderSetPipeline(pass, pipeline_handle);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);

        uint32_t vertex_count = (k_grid_cols - 1) * (k_grid_rows - 1) * 6;
        wgpuRenderPassEncoderDraw(pass, vertex_count, 1, 0, 0);
    });
}

void LobePass::draw_imgui_controls() {
    ImGui::SliderFloat("Roughness", &m_roughness, 0.01f, 1.0f);
    ImGui::SliderFloat("Metallic", &m_metallic, 0.0f, 1.0f);
    ImGui::SliderFloat("Scale", &m_scale, 0.1f, 5.0f);
    ImGui::SliderFloat("Light Azimuth", &m_light_azimuth_deg, -180.0f, 180.0f);
    ImGui::SliderFloat("Light Elevation", &m_light_elevation_deg, 0.0f, 90.0f);
}
