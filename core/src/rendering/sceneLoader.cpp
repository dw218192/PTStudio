#include <core/profiling.h>
#include <core/rendering/adapters/registry.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/primRange.h>

namespace pts::rendering {

namespace {

void sync_prim_impl(pxr::UsdPrim prim, RenderWorld& world, const webgpu::Device& device) {
    for (auto* adapter : k_scene_adapters()) {
        if (adapter->can_adapt(prim)) {
            adapter->sync(prim, world, device);
            break;
        }
    }
}

}  // namespace

void populate_from_stage(RenderWorld& world, const pxr::UsdStageRefPtr& stage,
                         const webgpu::Device& device) {
    PTS_ZONE_SCOPED;
    for (const auto& prim : pxr::UsdPrimRange(stage->GetPseudoRoot())) {
        sync_prim_impl(prim, world, device);
    }
}

void sync_prim(RenderWorld& world, const pxr::UsdStageRefPtr& stage, const webgpu::Device& device,
               const std::string& prim_path) {
    auto prim = stage->GetPrimAtPath(pxr::SdfPath(prim_path));
    if (!prim.IsValid()) {
        remove_prim(world, prim_path);
        return;
    }
    sync_prim_impl(prim, world, device);
}

void remove_prim(RenderWorld& world, const std::string& prim_path) {
    int obj_idx = world.find_object_by_prim(prim_path);
    if (obj_idx >= 0) {
        world.free_mesh_slot(world.objects[obj_idx].mesh_index);
        world.free_object_slot(static_cast<uint32_t>(obj_idx));
        ++world.mesh_version;
        return;
    }
    int light_idx = world.find_light_by_prim(prim_path);
    if (light_idx >= 0) {
        world.free_light_slot(static_cast<uint32_t>(light_idx));
        ++world.mesh_version;
    }
}

}  // namespace pts::rendering
