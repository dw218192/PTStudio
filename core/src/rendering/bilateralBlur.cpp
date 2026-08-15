#include <bilateral_blur_shader_metadata.h>
#include <core/profiling.h>
#include <core/rendering/bilateralBlur.h>
#include <core/rendering/webgpu/device.h>

#include <glm/glm.hpp>

namespace pts::rendering {

// Must match BilateralBlurUniforms in bilateral_blur.slang.
struct BilateralBlurUniforms {
    glm::vec2 texel_size;   // 0: 8
    float depth_threshold;  // 8: 4
    float _pad;             // 12: 4 -> total 16
};
static_assert(sizeof(BilateralBlurUniforms) == 16,
              "BilateralBlurUniforms must match shader std140 layout");

TextureDeclHandle add_bilateral_blur(FrameGraph& fg, const PassContext& ctx,
                                     const BilateralBlurParams& p) {
    PTS_ZONE_SCOPED;

    auto layout = fg.bind_group_layout(
        "bilateral_blur/layout",
        bilateral_blur_shader::create_bind_group_layout_0(ctx.device.handle()));

    // Pipeline keyed by caller label + output format so different callers /
    // formats get distinct pipelines. The label-uniqueness contract is the
    // caller's responsibility.
    std::string pipeline_name =
        p.debug_label + "/pipeline/" + std::to_string(static_cast<int>(p.output_format));
    auto* pipeline = fg.render_pipeline(pipeline_name)
                         .shader("core/generated/shaders/bilateral_blur.wgsl")
                         .color_format(p.output_format)
                         .cull_mode(WGPUCullMode_None)
                         .bind_group_layouts({layout})
                         .build();

    TextureDesc out_desc;
    out_desc.width = ctx.viewport_width;
    out_desc.height = ctx.viewport_height;
    out_desc.format = p.output_format;
    out_desc.clear_color = {1, 1, 1, 1};
    std::string out_label = p.debug_label + "/output";
    auto out_decl = fg.texture(out_label, out_desc);

    BufferDesc buf_desc;
    buf_desc.size = sizeof(BilateralBlurUniforms);
    buf_desc.usage =
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    std::string buf_label = p.debug_label + "/uniforms";
    auto uniform_decl = fg.buffer(buf_label, buf_desc);

    std::string desc_label = p.debug_label + "/desc";
    auto desc_decl = fg.descriptor(desc_label, layout)
                         .buffer(0, uniform_decl, 0, sizeof(BilateralBlurUniforms))
                         .texture(1, p.input)
                         .texture(2, p.depth)
                         .sampler(3, fg.sampler(WGPUSamplerBindingType_Filtering))
                         .sampler(4, fg.sampler(WGPUSamplerBindingType_NonFiltering))
                         .build();

    auto queue = ctx.queue;
    auto viewport_width = ctx.viewport_width;
    auto viewport_height = ctx.viewport_height;
    auto depth_threshold = p.depth_threshold;

    fg.add_pass(p.debug_label)
        .read(p.input)
        .read(p.depth)
        .color(out_decl)
        .execute([=](ExecuteContext& exec, WGPURenderPassEncoder pass) {
            auto uniform_buf = exec.get(uniform_decl).buffer;
            auto bind_group = exec.get(desc_decl).bind_group;

            BilateralBlurUniforms u{};
            u.texel_size = {1.0f / static_cast<float>(viewport_width),
                            1.0f / static_cast<float>(viewport_height)};
            u.depth_threshold = depth_threshold;
            wgpuQueueWriteBuffer(queue, uniform_buf, 0, &u, sizeof(u));

            wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

    return out_decl;
}

}  // namespace pts::rendering
