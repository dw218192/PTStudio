#include <core/diagnostics.h>
#include <core/rendering/renderWorld.h>

namespace pts::rendering {

namespace {

/// Generic slot allocator: pop from free list or append a default-constructed element.
template <typename T>
uint32_t alloc_slot(std::vector<T>& vec, std::vector<uint32_t>& free_list) {
    if (!free_list.empty()) {
        auto idx = free_list.back();
        free_list.pop_back();
        vec[idx] = T{};
        return idx;
    }
    vec.push_back(T{});
    return static_cast<uint32_t>(vec.size() - 1);
}

}  // namespace

uint32_t RenderWorld::alloc_object_slot() {
    return alloc_slot(objects, m_free_object_slots);
}

uint32_t RenderWorld::alloc_mesh_slot() {
    return alloc_slot(meshes, m_free_mesh_slots);
}

uint32_t RenderWorld::alloc_light_slot() {
    return alloc_slot(lights, m_free_light_slots);
}

void RenderWorld::free_object_slot(uint32_t i) {
    PRECONDITION(i < objects.size());
    PRECONDITION(objects[i].active);
    if (!objects[i].prim_path.empty()) {
        prim_slots.erase(objects[i].prim_path);
    }
    objects[i].active = false;
    objects[i].prim_path.clear();
    m_free_object_slots.push_back(i);
}

void RenderWorld::free_mesh_slot(uint32_t i) {
    PRECONDITION(i < meshes.size());
    meshes[i].vertex_buffer = {};
    meshes[i].index_buffer = {};
    meshes[i].index_count = 0;
    meshes[i].cpu_indices.clear();
    m_free_mesh_slots.push_back(i);
}

void RenderWorld::free_light_slot(uint32_t i) {
    PRECONDITION(i < lights.size());
    PRECONDITION(lights[i].active);
    if (!lights[i].prim_path.empty()) {
        prim_slots.erase(lights[i].prim_path);
    }
    lights[i].active = false;
    m_free_light_slots.push_back(i);
}

int RenderWorld::find_object_by_prim(std::string_view path) const {
    auto it = prim_slots.find(path);
    if (it == prim_slots.end() || it->second.kind != PrimSlot::Kind::Object) return -1;
    return static_cast<int>(it->second.index);
}

int RenderWorld::find_light_by_prim(std::string_view path) const {
    auto it = prim_slots.find(path);
    if (it == prim_slots.end() || it->second.kind != PrimSlot::Kind::Light) return -1;
    return static_cast<int>(it->second.index);
}

void RenderWorld::clear() {
    meshes.clear();
    objects.clear();
    materials.clear();
    lights.clear();
    material_cache.clear();
    prim_slots.clear();
    m_free_object_slots.clear();
    m_free_mesh_slots.clear();
    m_free_light_slots.clear();
}

}  // namespace pts::rendering
