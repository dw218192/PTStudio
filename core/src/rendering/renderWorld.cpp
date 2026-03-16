#include <core/diagnostics.h>
#include <core/rendering/adapterHelpers.h>
#include <core/rendering/renderWorld.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>

#include <algorithm>
#include <glm/glm.hpp>

namespace pts::rendering {

// --- SyncScope ---

SyncScope::SyncScope(RenderWorld& world) : m_world(world) {
}

SyncScope::~SyncScope() {
    ++m_world.m_mesh_version;
    ++m_world.m_light_version;
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
    return alloc_slot(m_world.m_objects, m_world.m_free_object_slots);
}

uint32_t SyncScope::alloc_mesh_slot() {
    return alloc_slot(m_world.m_meshes, m_world.m_free_mesh_slots);
}

uint32_t SyncScope::alloc_light_slot() {
    auto slot = alloc_slot(m_world.m_lights, m_world.m_free_light_slots);
    if (slot >= m_world.m_dirty_lights.size()) {
        m_world.m_dirty_lights.resize(m_world.m_lights.size(), 0);
    }
    m_world.m_dirty_lights[slot] = 1;
    return slot;
}

void SyncScope::free_object_slot(uint32_t i) {
    PRECONDITION(i < m_world.m_objects.size());
    PRECONDITION(m_world.m_objects[i].active);
    if (!m_world.m_objects[i].prim_path.empty()) {
        m_world.m_prim_slots.erase(m_world.m_objects[i].prim_path);
    }
    m_world.m_objects[i].active = false;
    m_world.m_objects[i].prim_path.clear();
    m_world.m_free_object_slots.push_back(i);
}

void SyncScope::free_mesh_slot(uint32_t i) {
    PRECONDITION(i < m_world.m_meshes.size());
    PRECONDITION(std::find(m_world.m_free_mesh_slots.begin(), m_world.m_free_mesh_slots.end(), i) ==
                 m_world.m_free_mesh_slots.end());
    m_world.m_meshes[i].vertex_buffer = {};
    m_world.m_meshes[i].index_buffer = {};
    m_world.m_meshes[i].index_count = 0;
    m_world.m_meshes[i].cpu_indices.clear();
    m_world.m_free_mesh_slots.push_back(i);
}

void SyncScope::free_light_slot(uint32_t i) {
    PRECONDITION(i < m_world.m_lights.size());
    PRECONDITION(m_world.m_lights[i].active);
    if (!m_world.m_lights[i].prim_path.empty()) {
        m_world.m_prim_slots.erase(m_world.m_lights[i].prim_path);
    }
    m_world.m_lights[i].active = false;
    m_world.m_dirty_lights[i] = 1;
    m_world.m_free_light_slots.push_back(i);
}

// --- SyncScope mutable accessors ---

RenderObject& SyncScope::object(uint32_t i) {
    return m_world.m_objects[i];
}

Mesh& SyncScope::mesh(uint32_t i) {
    return m_world.m_meshes[i];
}

Light& SyncScope::light(uint32_t i) {
    return m_world.m_lights[i];
}

Material& SyncScope::material(uint32_t i) {
    return m_world.m_materials[i];
}

std::vector<Material>& SyncScope::materials() {
    return m_world.m_materials;
}

std::unordered_map<std::string, uint32_t>& SyncScope::material_cache() {
    return m_world.m_material_cache;
}

void SyncScope::set_prim_slot(const std::string& path, PrimSlot slot) {
    m_world.m_prim_slots[path] = slot;
}

void SyncScope::mark_light_dirty(uint32_t i) {
    PRECONDITION(i < m_world.m_dirty_lights.size());
    m_world.m_dirty_lights[i] = 1;
}

void SyncScope::bump_light_version() {
    ++m_world.m_light_version;
}

// --- RenderWorld accessors ---

boost::span<const RenderObject> RenderWorld::get_objects() const {
    return {m_objects.data(), m_objects.size()};
}

boost::span<const Mesh> RenderWorld::get_meshes() const {
    return {m_meshes.data(), m_meshes.size()};
}

boost::span<const Light> RenderWorld::get_lights() const {
    return {m_lights.data(), m_lights.size()};
}

boost::span<const Material> RenderWorld::get_materials() const {
    return {m_materials.data(), m_materials.size()};
}

uint32_t RenderWorld::get_mesh_version() const {
    return m_mesh_version;
}

uint32_t RenderWorld::get_light_version() const {
    return m_light_version;
}

boost::span<const uint8_t> RenderWorld::get_dirty_lights() const {
    return {m_dirty_lights.data(), m_dirty_lights.size()};
}

void RenderWorld::clear_dirty_lights() {
    std::fill(m_dirty_lights.begin(), m_dirty_lights.end(), uint8_t{0});
}

// --- RenderWorld read-only + clear ---

int RenderWorld::find_object_by_prim(std::string_view path) const {
    auto it = m_prim_slots.find(path);
    if (it == m_prim_slots.end() || it->second.kind != PrimSlot::Kind::Object) return -1;
    return static_cast<int>(it->second.index);
}

int RenderWorld::find_light_by_prim(std::string_view path) const {
    auto it = m_prim_slots.find(path);
    if (it == m_prim_slots.end() || it->second.kind != PrimSlot::Kind::Light) return -1;
    return static_cast<int>(it->second.index);
}

void RenderWorld::clear() {
    m_meshes.clear();
    m_objects.clear();
    m_materials.clear();
    m_lights.clear();
    m_dirty_lights.clear();
    m_material_cache.clear();
    m_prim_slots.clear();
    m_free_object_slots.clear();
    m_free_mesh_slots.clear();
    m_free_light_slots.clear();
}

// --- update_transforms ---

void RenderWorld::update_transforms(const pxr::UsdStageRefPtr& stage,
                                    const std::vector<pxr::SdfPath>& dirty_paths) {
    for (const auto& dirty_path : dirty_paths) {
        for (const auto& [path, slot] : m_prim_slots) {
            auto slot_path = pxr::SdfPath(path);
            if (!slot_path.HasPrefix(dirty_path)) continue;

            auto prim = stage->GetPrimAtPath(slot_path);
            if (!prim.IsValid()) continue;

            auto xf = compute_world_transform(prim);

            if (slot.kind == PrimSlot::Kind::Object) {
                m_objects[slot.index].transform = xf;
            } else {
                auto& light = m_lights[slot.index];
                light.transform = xf;
                if (light.type == Light::Type::Distant) {
                    glm::vec4 local_dir(0.0f, 0.0f, -1.0f, 0.0f);
                    light.direction = glm::normalize(glm::vec3(xf * local_dir));
                }
                m_dirty_lights[slot.index] = 1;
                ++m_light_version;
            }
        }
    }
}

}  // namespace pts::rendering
