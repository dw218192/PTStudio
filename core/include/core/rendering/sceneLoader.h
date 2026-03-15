#pragma once

#include <core/rendering/webgpu/device.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>

namespace pts::rendering {

struct RenderWorld;

void populate_from_stage(RenderWorld& world, const pxr::UsdStageRefPtr& stage,
                         const webgpu::Device& device);

void sync_prim(RenderWorld& world, const pxr::UsdStageRefPtr& stage, const webgpu::Device& device,
               const pxr::SdfPath& prim_path);

void remove_prim(RenderWorld& world, const pxr::SdfPath& prim_path);

}  // namespace pts::rendering
