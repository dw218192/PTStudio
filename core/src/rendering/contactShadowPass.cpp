#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/contactShadowPass.h>
#include <core/rendering/fallbackPool.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/gbufferPass.h>
#include <core/rendering/passContext.h>
#include <core/rendering/shaderLoader.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <imgui.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

namespace pts::rendering {

// Must match ContactShadowUniforms in contact_shadow.slang (std140 layout).
struct ContactShadowUniforms {
    glm::mat4 projection;      // 0:   64
    glm::mat4 inv_projection;  // 64:  64
    glm::mat4 view;            // 128: 64
    glm::vec2 viewport_size;   // 192: 8
    float max_distance;        // 200: 4
    float thickness;           // 204: 4
    float normal_offset;       // 208: 4
    int32_t step_count;        // 212: 4
    uint32_t light_count;      // 216: 4
    uint32_t _pad;             // 220: 4 → total 224
};
static_assert(sizeof(ContactShadowUniforms) == 224,
              "ContactShadowUniforms must match shader std140 layout");

ContactShadowPass::ContactShadowPass(const ShaderLoader& sl, const GBufferPass& gbuf)
    : IPass(sl), m_gbuf(&gbuf) {
}

ContactShadowPass::~ContactShadowPass() {
    release_raw_handles();
}

void ContactShadowPass::release_raw_handles() {
    if (auto* ready = std::get_if<Ready>(&m_state)) {
        ready->internal_layout.release();
        ready->output_layout.release();
    }
}

auto ContactShadowPass::is_ready() const noexcept -> bool {
    return std::holds_alternative<Ready>(m_state);
}

static constexpr IPass::DebugTarget k_debug_targets[] = {
    {"Contact Shadow", "contact_shadow"},
};

auto ContactShadowPass::debug_targets() const noexcept -> std::pair<const DebugTarget*, uint32_t> {
    return {k_debug_targets, m_enabled ? 1u : 0u};
}

void ContactShadowPass::do_setup(const webgpu::Device& device) {
    release_raw_handles();

    auto shader_src = get_shader_loader().load("core/generated/shaders/contact_shadow.wgsl");
    auto shader = device.create_shader_module_from_source(shader_src);

    // ── Bind group layout ──
    // GBuffer consumer slots: 0=depth_tex, 1=depth_sampler, 2=normals_tex, 3=normals_sampler
    // ContactShadow-specific: 4=uniforms, 5=lights
    PRECONDITION(m_gbuf->is_ready());
    auto gbuf_slots = m_gbuf->consumer_output_slots();
    std::vector<OutputSlot> slots;
    slots.insert(slots.end(), gbuf_slots.begin(), gbuf_slots.end());
    slots.push_back(OutputSlot::uniform(sizeof(ContactShadowUniforms)));
    slots.push_back(OutputSlot::storage());
    auto internal_layout = create_output_layout(device, slots);

    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &internal_layout.layout;
    auto pl = wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

    auto pipeline = webgpu::RenderPipelineBuilder(device)
                        .shader(shader)
                        .color_format(WGPUTextureFormat_R8Unorm)
                        .cull_mode(WGPUCullMode_None)
                        .pipeline_layout(pl)
                        .build();
    wgpuPipelineLayoutRelease(pl);

    // Consumer output layout: slot 0 = contact shadow texture, slot 1 = sampler
    auto st = OutputSlot::sampled_texture(WGPUTextureFormat_R8Unorm);
    auto output_layout = create_output_layout(device, {st[0], st[1]});

    m_state = Ready{
        std::move(shader),
        std::move(pipeline),
        std::move(internal_layout),
        std::move(output_layout),
    };
}

WGPUBindGroupLayout ContactShadowPass::consumer_layout() const {
    PRECONDITION(is_ready());
    return std::get<Ready>(m_state).output_layout.layout;
}

ContactShadowPass::Outputs ContactShadowPass::add_to_frame_graph(FrameGraph& fg,
                                                                 const PassContext& ctx,
                                                                 const Inputs& in,
                                                                 FallbackPool& fallbacks) {
    PTS_ZONE_SCOPED;
    PRECONDITION(is_ready());
    auto& ready = std::get<Ready>(m_state);
    auto& ol = ready.output_layout;

    if (!m_enabled) {
        auto consumer = ol.build(fg, this, {TextureHandle{}}, fallbacks, "consumer_desc");
        return {{}, consumer};
    }

    // ── Frame graph resources ──
    TextureDesc cs_desc;
    cs_desc.width = ctx.viewport_width;
    cs_desc.height = ctx.viewport_height;
    cs_desc.format = WGPUTextureFormat_R8Unorm;
    cs_desc.clear_color = {1, 1, 1, 1};
    auto cs_handle = create_texture(fg, cs_desc, "contact_shadow");

    BufferDesc uniform_buf_desc;
    uniform_buf_desc.size = sizeof(ContactShadowUniforms);
    uniform_buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    auto uniform_buf_handle = create_buffer(fg, uniform_buf_desc, "cs_uniforms");

    // Non-sampler resources in slot order: depth(0), normals(2), uniforms(4), lights(5)
    auto bg_handle =
        ready.internal_layout.build(fg, this,
                                    {TextureHandle{in.depth}, TextureHandle{in.normals},
                                     BufferHandle{uniform_buf_handle}, in.light_buffer},
                                    fallbacks, "cs_bg");

    // Consumer descriptor: managed CS texture + sampler
    auto consumer = ol.build(fg, this, {TextureHandle{cs_handle}}, fallbacks, "consumer_desc");

    // Capture scalars for lambda
    auto* pipeline = ready.pipeline.handle();
    auto queue = ctx.queue;
    auto proj_matrix = ctx.proj_matrix;
    auto view_matrix = ctx.view_matrix;
    auto viewport_width = ctx.viewport_width;
    auto viewport_height = ctx.viewport_height;
    auto light_count = ctx.world.gpu_light_count();
    auto max_distance = m_max_distance;
    auto thickness = m_thickness;
    auto normal_offset = m_normal_offset;
    auto step_count = m_step_count;

    fg.add_pass("contact_shadow_gen")
        .read(in.depth)
        .read(in.normals)
        .color(cs_handle)
        .execute([=, &fg](WGPURenderPassEncoder pass) {
            auto uniform_buf = fg.get_buffer_ref(uniform_buf_handle).handle();
            auto bg = fg.get_descriptor_ref(bg_handle).handle();

            ContactShadowUniforms uniforms{};
            uniforms.projection = proj_matrix;
            uniforms.inv_projection = glm::inverse(proj_matrix);
            uniforms.view = view_matrix;
            uniforms.viewport_size = {
                static_cast<float>(viewport_width),
                static_cast<float>(viewport_height),
            };
            uniforms.max_distance = max_distance;
            uniforms.thickness = thickness;
            uniforms.normal_offset = normal_offset;
            uniforms.step_count = step_count;
            uniforms.light_count = light_count;
            wgpuQueueWriteBuffer(queue, uniform_buf, 0, &uniforms, sizeof(uniforms));

            wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

    return {cs_handle, consumer};
}

void ContactShadowPass::draw_imgui() {
    ImGui::Checkbox("Enabled", &m_enabled);
    ImGui::SliderFloat("Max Distance", &m_max_distance, 0.01f, 2.0f);
    ImGui::SliderFloat("Thickness", &m_thickness, 0.001f, 0.2f);
    ImGui::SliderFloat("Normal Offset", &m_normal_offset, 0.0f, 0.1f);
    ImGui::SliderInt("Step Count", &m_step_count, 4, 64);
}

}  // namespace pts::rendering
