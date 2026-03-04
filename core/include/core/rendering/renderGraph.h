#pragma once

#include <core/rendering/webgpu/webgpu.h>

#include <functional>
#include <string>
#include <vector>

namespace pts::rendering {

struct RenderWorld;

struct PassDesc {
    std::string name;
    WGPUTextureView color_target;
    WGPUTextureFormat color_format;
    WGPUColor clear_color = {0.1, 0.1, 0.1, 1.0};
    WGPUTextureView depth_target = nullptr;
};

class RenderGraph {
   public:
    using PassCallback = std::function<void(WGPURenderPassEncoder, const RenderWorld&)>;

    void add_pass(PassDesc desc, PassCallback callback);
    void execute(WGPUCommandEncoder encoder, const RenderWorld& world);
    void clear();

   private:
    struct Pass {
        PassDesc desc;
        PassCallback callback;
    };
    std::vector<Pass> m_passes;
};

}  // namespace pts::rendering
