#include <core/diagnostics.h>
#include <core/rendering/adapterHelpers.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/webgpu/device.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>

#include <algorithm>
#include <glm/glm.hpp>

namespace pts::rendering {

// --- to_light ---

Light to_light(const LightSlot& slot) {
    Light l{};
    l.type = static_cast<uint32_t>(slot.type);
    l.color = slot.color;
    l.intensity = slot.intensity;
    l.radius = slot.radius;
    l.width = slot.width;
    l.height = slot.height;
    l.angle = slot.angle;

    if (slot.type == LightSlot::Type::Distant) {
        l.direction_or_pos = slot.direction;
    } else {
        l.direction_or_pos = glm::vec3(slot.transform[3]);
    }
    return l;
}

// --- SyncScope ---

SyncScope::SyncScope(RenderWorld& world) : m_world(world) {
}

SyncScope::~SyncScope() {
    ++m_world.m_mesh_version;
    ++m_world.m_light_version;
    ++m_world.m_material_version;
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
    m_world.m_meshes[i].cpu_vertices.clear();
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

ObjectSlot& SyncScope::object(uint32_t i) {
    return m_world.m_objects[i];
}

Mesh& SyncScope::mesh(uint32_t i) {
    return m_world.m_meshes[i];
}

LightSlot& SyncScope::light(uint32_t i) {
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

boost::span<const ObjectSlot> RenderWorld::get_objects() const {
    return {m_objects.data(), m_objects.size()};
}

boost::span<const Mesh> RenderWorld::get_meshes() const {
    return {m_meshes.data(), m_meshes.size()};
}

boost::span<const LightSlot> RenderWorld::get_lights() const {
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

uint32_t RenderWorld::get_material_version() const {
    return m_material_version;
}

const webgpu::Buffer& RenderWorld::light_buffer() const {
    return m_gpu_light_buffer;
}

const webgpu::Buffer& RenderWorld::material_buffer() const {
    return m_gpu_material_buffer;
}

uint32_t RenderWorld::gpu_light_count() const {
    return m_gpu_light_count;
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

// --- GPU buffer upload ---

namespace {
constexpr std::size_t k_min_material_buffer_size = sizeof(Material);  // 32 bytes
constexpr std::size_t k_min_light_buffer_size = sizeof(Light);        // 48 bytes
}  // namespace

void RenderWorld::prepare_gpu_buffers(const webgpu::Device& device, WGPUQueue queue) {
    // --- Materials ---
    if (m_material_version != m_cached_material_version) {
        auto material_count = static_cast<uint32_t>(m_materials.size());
        auto required_size = std::max(k_min_material_buffer_size,
                                      static_cast<std::size_t>(material_count) * sizeof(Material));

        if (required_size > m_gpu_material_buffer.size()) {
            m_gpu_material_buffer = device.create_buffer(
                required_size,
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
        }

        if (material_count > 0) {
            wgpuQueueWriteBuffer(queue, m_gpu_material_buffer.handle(), 0, m_materials.data(),
                                 material_count * sizeof(Material));
        }

        m_cached_material_version = m_material_version;
    }

    // --- Lights ---
    if (m_light_version != m_cached_light_version) {
        // Full rebuild: collect active lights into GPU format
        std::vector<Light> gpu_lights;
        for (const auto& slot : m_lights) {
            if (!slot.active) continue;
            gpu_lights.push_back(to_light(slot));
        }

        // Default fallback: single distant light when scene has no lights
        if (gpu_lights.empty()) {
            Light def{};
            def.type = 0;
            def.direction_or_pos = glm::normalize(glm::vec3(0.3f, -1.0f, 0.5f));
            def.color = {1.0f, 0.95f, 0.9f};
            def.intensity = 1.0f;
            gpu_lights.push_back(def);
        }

        auto buf_size = std::max(k_min_light_buffer_size, gpu_lights.size() * sizeof(Light));

        if (!m_gpu_light_buffer.is_valid() || m_gpu_light_buffer.size() < buf_size) {
            m_gpu_light_buffer = device.create_buffer(
                buf_size,
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
        }

        wgpuQueueWriteBuffer(queue, m_gpu_light_buffer.handle(), 0, gpu_lights.data(),
                             gpu_lights.size() * sizeof(Light));
        m_gpu_light_count = static_cast<uint32_t>(gpu_lights.size());
        m_cached_light_version = m_light_version;
        clear_dirty_lights();
    } else if (!m_dirty_lights.empty()) {
        // Partial update: write only dirty slots
        uint32_t gpu_idx = 0;
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_lights.size()); ++i) {
            if (!m_lights[i].active) continue;
            if (i < static_cast<uint32_t>(m_dirty_lights.size()) && m_dirty_lights[i]) {
                auto gl = to_light(m_lights[i]);
                wgpuQueueWriteBuffer(queue, m_gpu_light_buffer.handle(), gpu_idx * sizeof(Light),
                                     &gl, sizeof(Light));
            }
            ++gpu_idx;
        }
        clear_dirty_lights();
    }
}

void RenderWorld::upload_all_meshes(const webgpu::Device& device) {
    for (auto& mesh : m_meshes) {
        if (mesh.cpu_vertices.empty()) continue;

        PRECONDITION(!mesh.cpu_indices.empty());

        mesh.vertex_buffer = device.create_buffer(
            mesh.cpu_vertices.size() * sizeof(Vertex),
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst));
        wgpuQueueWriteBuffer(device.queue(), mesh.vertex_buffer.handle(), 0,
                             mesh.cpu_vertices.data(),
                             mesh.cpu_vertices.size() * sizeof(Vertex));

        mesh.index_buffer = device.create_buffer(
            mesh.cpu_indices.size() * sizeof(uint32_t),
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst));
        wgpuQueueWriteBuffer(device.queue(), mesh.index_buffer.handle(), 0,
                             mesh.cpu_indices.data(),
                             mesh.cpu_indices.size() * sizeof(uint32_t));

        mesh.index_count = static_cast<uint32_t>(mesh.cpu_indices.size());
        mesh.cpu_vertices.clear();
        mesh.cpu_vertices.shrink_to_fit();
        ++mesh.version;
    }
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
    m_gpu_light_buffer = {};
    m_gpu_material_buffer = {};
    m_gpu_light_count = 0;
    m_cached_light_version = UINT32_MAX;
    m_cached_material_version = UINT32_MAX;
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
                if (light.type == LightSlot::Type::Distant) {
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
