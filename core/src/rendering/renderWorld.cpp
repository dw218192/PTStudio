#include <core/diagnostics.h>
#include <core/rendering/renderWorld.h>

namespace pts::rendering {

uint32_t RenderWorld::alloc_object_slot() {
    if (!free_object_slots.empty()) {
        auto idx = free_object_slots.back();
        free_object_slots.pop_back();
        objects[idx] = RenderObject{};
        return idx;
    }
    objects.push_back(RenderObject{});
    return static_cast<uint32_t>(objects.size() - 1);
}

uint32_t RenderWorld::alloc_mesh_slot() {
    if (!free_mesh_slots.empty()) {
        auto idx = free_mesh_slots.back();
        free_mesh_slots.pop_back();
        meshes[idx] = Mesh{};
        return idx;
    }
    meshes.push_back(Mesh{});
    return static_cast<uint32_t>(meshes.size() - 1);
}

uint32_t RenderWorld::alloc_light_slot() {
    if (!free_light_slots.empty()) {
        auto idx = free_light_slots.back();
        free_light_slots.pop_back();
        lights[idx] = Light{};
        return idx;
    }
    lights.push_back(Light{});
    return static_cast<uint32_t>(lights.size() - 1);
}

void RenderWorld::free_object_slot(uint32_t i) {
    PRECONDITION(i < objects.size());
    PRECONDITION(objects[i].active);
    if (!objects[i].prim_path.empty()) {
        prim_to_object.erase(objects[i].prim_path);
    }
    objects[i].active = false;
    objects[i].prim_path.clear();
    free_object_slots.push_back(i);
}

void RenderWorld::free_mesh_slot(uint32_t i) {
    PRECONDITION(i < meshes.size());
    meshes[i].vertex_buffer = {};
    meshes[i].index_buffer = {};
    meshes[i].index_count = 0;
    meshes[i].cpu_indices.clear();
    free_mesh_slots.push_back(i);
}

void RenderWorld::free_light_slot(uint32_t i) {
    PRECONDITION(i < lights.size());
    PRECONDITION(lights[i].active);
    if (!lights[i].prim_path.empty()) {
        prim_to_light.erase(lights[i].prim_path);
    }
    lights[i].active = false;
    free_light_slots.push_back(i);
}

int RenderWorld::find_object_by_prim(const std::string& path) const {
    auto it = prim_to_object.find(path);
    if (it == prim_to_object.end()) return -1;
    return static_cast<int>(it->second);
}

int RenderWorld::find_light_by_prim(const std::string& path) const {
    auto it = prim_to_light.find(path);
    if (it == prim_to_light.end()) return -1;
    return static_cast<int>(it->second);
}

void RenderWorld::clear() {
    meshes.clear();
    objects.clear();
    materials.clear();
    lights.clear();
    material_cache.clear();
    prim_to_object.clear();
    prim_to_light.clear();
    free_object_slots.clear();
    free_mesh_slots.clear();
    free_light_slots.clear();
}

}  // namespace pts::rendering
