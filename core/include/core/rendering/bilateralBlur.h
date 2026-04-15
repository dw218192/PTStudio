#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/passContext.h>
#include <core/rendering/webgpu/webgpu.h>

#include <string>

namespace pts::rendering {

/// Depth-aware 4x4 bilateral blur helper. Reads the `.r` channel of `input`,
/// rejecting neighbor samples whose linear depth differs from the center by
/// more than `depth_threshold`. Produces a texture of the same size as the
/// viewport in `output_format`.
///
/// Pipelines and layouts are cached in the frame graph keyed by the shader
/// module + `debug_label`, so each caller should pass a unique label.
struct BilateralBlurParams {
    TextureDeclHandle input;
    TextureDeclHandle depth;
    WGPUTextureFormat output_format = WGPUTextureFormat_R8Unorm;
    float depth_threshold = 0.001f;
    std::string debug_label = "bilateral_blur";
};

TextureDeclHandle add_bilateral_blur(FrameGraph& fg, const PassContext& ctx,
                                     const BilateralBlurParams& p);

}  // namespace pts::rendering
