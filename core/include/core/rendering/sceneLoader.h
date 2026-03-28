#pragma once

#include <core/rendering/upAxis.h>
#include <core/worker.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>

namespace pts::rendering {

struct RenderWorld;
class SyncScope;

struct StageSettings {
    float meters_per_unit = 0.01f;  // USD default = centimeters
    UpAxis up_axis = UpAxis::Y;
};

/// Read metersPerUnit and upAxis from USD stage metadata.
StageSettings read_stage_settings(const pxr::UsdStageRefPtr& stage);

void populate_from_stage(RenderWorld& world, const pxr::UsdStageRefPtr& stage);

void sync_prim(SyncScope& scope, const pxr::UsdStageRefPtr& stage, const pxr::SdfPath& prim_path);

void remove_prim(SyncScope& scope, const pxr::SdfPath& prim_path);

/// Build a fresh RenderWorld with CPU data only (no GPU buffers).
/// Reports progress; suitable for calling from a background thread.
RenderWorld populate_from_stage(const pxr::UsdStageRefPtr& stage, TaskProgress& progress);

}  // namespace pts::rendering
