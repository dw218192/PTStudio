#include <core/backgroundTask.h>
#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/registry.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <pxr/usd/usd/primRange.h>

#include <glm/glm.hpp>

namespace pts::rendering {

namespace {

void sync_prim_impl(pxr::UsdPrim prim, SyncScope& scope) {
    for (auto* adapter : k_scene_adapters()) {
        if (adapter->can_adapt(prim)) {
            adapter->sync(prim, scope);
            return;
        }
    }
    // No adapter handles this prim — remove any stale entry from a prior sync
    // (e.g. prim type changed from mesh to something unsupported).
    remove_prim(scope, prim.GetPath());
}

}  // namespace

void populate_from_stage(RenderWorld& world, const pxr::UsdStageRefPtr& stage) {
    PTS_ZONE_SCOPED;
    auto scope = world.begin_sync();
    for (const auto& prim : pxr::UsdPrimRange(stage->GetPseudoRoot())) {
        sync_prim_impl(prim, scope);
    }
}

void sync_prim(SyncScope& scope, const pxr::UsdStageRefPtr& stage, const pxr::SdfPath& prim_path) {
    auto prim = stage->GetPrimAtPath(prim_path);
    if (!prim.IsValid()) {
        remove_prim(scope, prim_path);
        return;
    }
    sync_prim_impl(prim, scope);
}

void remove_prim(SyncScope& scope, const pxr::SdfPath& prim_path) {
    auto& world = scope.world();
    int obj_idx = world.find_object_by_prim(prim_path);
    if (obj_idx >= 0) {
        scope.free_mesh_slot(scope.object(static_cast<uint32_t>(obj_idx))->mesh_index);
        scope.free_object_slot(static_cast<uint32_t>(obj_idx));
        return;
    }
    int light_idx = world.find_light_by_prim(prim_path);
    if (light_idx >= 0) {
        scope.free_light_slot(static_cast<uint32_t>(light_idx));
        return;
    }
    int cam_idx = world.find_camera_by_prim(prim_path);
    if (cam_idx >= 0) {
        scope.free_camera_slot(static_cast<uint32_t>(cam_idx));
    }
}

RenderWorld populate_from_stage(const pxr::UsdStageRefPtr& stage, TaskProgress& progress) {
    PTS_ZONE_SCOPED;

    size_t total = 0;
    for (const auto& prim : pxr::UsdPrimRange(stage->GetPseudoRoot())) {
        PTS_UNUSED(prim);
        ++total;
    }

    RenderWorld world;
    auto scope = world.begin_sync();
    size_t current = 0;
    for (const auto& prim : pxr::UsdPrimRange(stage->GetPseudoRoot())) {
        sync_prim_impl(prim, scope);
        ++current;
        progress.set_progress(static_cast<float>(current) / static_cast<float>(total));
        progress.set_status(prim.GetPath().GetText());
    }

    return world;
}

}  // namespace pts::rendering
