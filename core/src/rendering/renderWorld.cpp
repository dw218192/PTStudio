#include <core/diagnostics.h>
#include <core/rendering/renderWorld.h>

#include <algorithm>

namespace pts::rendering {

// --- SyncScope ---

SyncScope::SyncScope(RenderWorld& world) : m_world(world) {
}

SyncScope::~SyncScope() {
    ++m_world.mesh_version;
    ++m_world.light_version;
}

SyncScope RenderWorld::begin_sync() {
    return SyncScope(*this);
}

// --- Slot allocation (via SyncScope) ---

namespace {

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

uint32_t SyncScope::alloc_object_slot() {
    return alloc_slot(m_world.objects, m_world.m_free_object_slots);
}

uint32_t SyncScope::alloc_mesh_slot() {
    return alloc_slot(m_world.meshes, m_world.m_free_mesh_slots);
}

uint32_t SyncScope::alloc_light_slot() {
    return alloc_slot(m_world.lights, m_world.m_free_light_slots);
}

void SyncScope::free_object_slot(uint32_t i) {
    PRECONDITION(i < m_world.objects.size());
    PRECONDITION(m_world.objects[i].active);
    if (!m_world.objects[i].prim_path.empty()) {
        m_world.prim_slots.erase(m_world.objects[i].prim_path);
    }
    m_world.objects[i].active = false;
    m_world.objects[i].prim_path.clear();
    m_world.m_free_object_slots.push_back(i);
}

void SyncScope::free_mesh_slot(uint32_t i) {
    PRECONDITION(i < m_world.meshes.size());
    PRECONDITION(std::find(m_world.m_free_mesh_slots.begin(), m_world.m_free_mesh_slots.end(), i) ==
                 m_world.m_free_mesh_slots.end());
    m_world.meshes[i].vertex_buffer = {};
    m_world.meshes[i].index_buffer = {};
    m_world.meshes[i].index_count = 0;
    m_world.meshes[i].cpu_indices.clear();
    m_world.m_free_mesh_slots.push_back(i);
}

void SyncScope::free_light_slot(uint32_t i) {
    PRECONDITION(i < m_world.lights.size());
    PRECONDITION(m_world.lights[i].active);
    if (!m_world.lights[i].prim_path.empty()) {
        m_world.prim_slots.erase(m_world.lights[i].prim_path);
    }
    m_world.lights[i].active = false;
    m_world.m_free_light_slots.push_back(i);
}

// --- RenderWorld read-only + clear ---

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
