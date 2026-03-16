#pragma once

#include <core/rendering/webgpu/device.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>

namespace pts::rendering {

struct RenderWorld;
class SyncScope;

void populate_from_stage(RenderWorld& world, const pxr::UsdStageRefPtr& stage,
                         const webgpu::Device& device);

void sync_prim(SyncScope& scope, const pxr::UsdStageRefPtr& stage, const webgpu::Device& device,
               const pxr::SdfPath& prim_path);

void remove_prim(SyncScope& scope, const pxr::SdfPath& prim_path);

/// Lightweight xform-only update: recomputes world transforms for all
/// synced prims at or under the given paths. Does not re-upload meshes.
void update_transforms(RenderWorld& world, const pxr::UsdStageRefPtr& stage,
                       const std::vector<pxr::SdfPath>& dirty_paths);

}  // namespace pts::rendering
