#pragma once

#include <core/backgroundTask.h>
#include <core/rendering/webgpu/device.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>

namespace pts::rendering {

struct RenderWorld;
class SyncScope;

void populate_from_stage(RenderWorld& world, const pxr::UsdStageRefPtr& stage,
                         const webgpu::Device* device);

void sync_prim(SyncScope& scope, const pxr::UsdStageRefPtr& stage, const webgpu::Device* device,
               const pxr::SdfPath& prim_path);

void remove_prim(SyncScope& scope, const pxr::SdfPath& prim_path);

/// Build a RenderWorld with CPU data only (no GPU buffers).
/// Suitable for calling from a background thread on a freshly-opened stage.
RenderWorld populate_from_stage_cpu(const pxr::UsdStageRefPtr& stage, TaskProgress& progress);

}  // namespace pts::rendering
