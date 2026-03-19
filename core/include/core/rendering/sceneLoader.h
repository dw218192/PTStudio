#pragma once

#include <core/backgroundTask.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>

namespace pts::rendering {

struct RenderWorld;
class SyncScope;

void populate_from_stage(RenderWorld& world, const pxr::UsdStageRefPtr& stage);

void sync_prim(SyncScope& scope, const pxr::UsdStageRefPtr& stage, const pxr::SdfPath& prim_path);

void remove_prim(SyncScope& scope, const pxr::SdfPath& prim_path);

/// Build a fresh RenderWorld with CPU data only (no GPU buffers).
/// Reports progress; suitable for calling from a background thread.
RenderWorld populate_from_stage(const pxr::UsdStageRefPtr& stage, TaskProgress& progress);

}  // namespace pts::rendering
