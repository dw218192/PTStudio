#include <core/profiling.h>
#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/registry.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <pxr/usd/usd/primRange.h>

#include <glm/glm.hpp>

namespace pts::rendering {

namespace {

void sync_prim_impl(pxr::UsdPrim prim, SyncScope& scope, const webgpu::Device& device) {
    for (auto* adapter : k_scene_adapters()) {
        if (adapter->can_adapt(prim)) {
            adapter->sync(prim, scope, device);
            return;
        }
    }
    // No adapter handles this prim — remove any stale entry from a prior sync
    // (e.g. prim type changed from mesh to something unsupported).
    remove_prim(scope, prim.GetPath());
}

}  // namespace

void populate_from_stage(RenderWorld& world, const pxr::UsdStageRefPtr& stage,
                         const webgpu::Device& device) {
    PTS_ZONE_SCOPED;
    auto scope = world.begin_sync();
    for (const auto& prim : pxr::UsdPrimRange(stage->GetPseudoRoot())) {
        sync_prim_impl(prim, scope, device);
    }
}

void sync_prim(SyncScope& scope, const pxr::UsdStageRefPtr& stage, const webgpu::Device& device,
               const pxr::SdfPath& prim_path) {
    auto prim = stage->GetPrimAtPath(prim_path);
    if (!prim.IsValid()) {
        remove_prim(scope, prim_path);
        return;
    }
    sync_prim_impl(prim, scope, device);
}

void remove_prim(SyncScope& scope, const pxr::SdfPath& prim_path) {
    auto& world = scope.world();
    auto path_text = prim_path.GetText();
    int obj_idx = world.find_object_by_prim(path_text);
    if (obj_idx >= 0) {
        scope.free_mesh_slot(world.objects[obj_idx].mesh_index);
        scope.free_object_slot(static_cast<uint32_t>(obj_idx));
        return;
    }
    int light_idx = world.find_light_by_prim(path_text);
    if (light_idx >= 0) {
        scope.free_light_slot(static_cast<uint32_t>(light_idx));
    }
}

void update_transforms(RenderWorld& world, const pxr::UsdStageRefPtr& stage,
                       const std::vector<pxr::SdfPath>& dirty_paths) {
    for (const auto& dirty_path : dirty_paths) {
        for (const auto& [path, slot] : world.prim_slots) {
            auto slot_path = pxr::SdfPath(path);
            if (!slot_path.HasPrefix(dirty_path)) continue;

            auto prim = stage->GetPrimAtPath(slot_path);
            if (!prim.IsValid()) continue;

            auto xf = compute_world_transform(prim);

            if (slot.kind == PrimSlot::Kind::Object) {
                world.objects[slot.index].transform = xf;
            } else {
                auto& light = world.lights[slot.index];
                light.transform = xf;
                if (light.type == Light::Type::Distant) {
                    glm::vec4 local_dir(0.0f, 0.0f, -1.0f, 0.0f);
                    light.direction = glm::normalize(glm::vec3(xf * local_dir));
                }
                ++world.light_version;
            }
        }
    }
}

}  // namespace pts::rendering
