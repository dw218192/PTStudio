#pragma once

#include <core/rendering/webgpu/device.h>
#include <pxr/usd/usd/stage.h>

namespace pts::rendering {

struct RenderWorld;

void populate_from_stage(RenderWorld& world, const pxr::UsdStageRefPtr& stage,
                         const webgpu::Device& device);


}  // namespace pts::rendering
