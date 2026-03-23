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

Light to_light(const LightData& slot) {
    Light l{};
    l.type = static_cast<uint32_t>(slot.type);
    l.color = slot.color;
    l.intensity = std::max(slot.intensity, 0.0f);
    l.radius = slot.radius;
    l.angle = slot.angle;

    if (slot.type == LightData::Type::Distant) {
        l.direction_or_pos = slot.direction;
        l.right = glm::vec3(0.0f);
        l.up = glm::vec3(0.0f);
    } else {
        l.direction_or_pos = glm::vec3(slot.transform[3]);
        if (slot.type == LightData::Type::Rect) {
            l.right = glm::normalize(glm::vec3(slot.transform[0])) * (slot.width / 2.0f);
            l.up = glm::normalize(glm::vec3(slot.transform[1])) * (slot.height / 2.0f);
        } else if (slot.type == LightData::Type::Disk) {
            l.right = glm::normalize(glm::vec3(slot.transform[0])) * slot.radius;
            l.up = glm::normalize(glm::vec3(slot.transform[1])) * slot.radius;
        } else {
            l.right = glm::vec3(0.0f);
            l.up = glm::vec3(0.0f);
        }
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

uint32_t SyncScope::alloc_object_slot() {
    return m_world.m_objects.alloc();
}

uint32_t SyncScope::alloc_mesh_slot() {
    return m_world.m_meshes.alloc();
}

uint32_t SyncScope::alloc_light_slot() {
    return m_world.m_lights.alloc();
}

void SyncScope::free_object_slot(uint32_t i) {
    const auto& prim_path = m_world.m_objects[i].get_prim_path();
    if (!prim_path.IsEmpty()) {
        auto it = m_world.m_prim_slots.find(prim_path);
        if (it != m_world.m_prim_slots.end()) m_world.m_prim_slots.erase(it);
    }
    m_world.m_objects.set_prim_path(i, pxr::SdfPath());
    m_world.m_objects.free(i);
}

void SyncScope::free_mesh_slot(uint32_t i) {
    // Clear mesh resources before freeing
    {
        auto w = m_world.m_meshes.write(i);
        w->vertex_buffer = {};
        w->index_buffer = {};
        w->index_count = 0;
        w->cpu_indices.clear();
        w->cpu_vertices.clear();
    }
    m_world.m_meshes.free(i);
}

void SyncScope::free_light_slot(uint32_t i) {
    const auto& prim_path = m_world.m_lights[i].get_prim_path();
    if (!prim_path.IsEmpty()) {
        auto it = m_world.m_prim_slots.find(prim_path);
        if (it != m_world.m_prim_slots.end()) m_world.m_prim_slots.erase(it);
    }
    m_world.m_lights.set_prim_path(i, pxr::SdfPath());
    m_world.m_lights.free(i);
}

uint32_t SyncScope::alloc_camera_slot() {
    return m_world.m_cameras.alloc();
}

void SyncScope::free_camera_slot(uint32_t i) {
    const auto& prim_path = m_world.m_cameras[i].get_prim_path();
    if (!prim_path.IsEmpty()) {
        auto it = m_world.m_prim_slots.find(prim_path);
        if (it != m_world.m_prim_slots.end()) m_world.m_prim_slots.erase(it);
    }
    m_world.m_cameras.set_prim_path(i, pxr::SdfPath());
    m_world.m_cameras.free(i);
}

// --- SyncScope accessors ---

Slot<ObjectData>::WriteGuard SyncScope::write_object(uint32_t i) {
    return m_world.m_objects.write(i);
}

Slot<MeshData>::WriteGuard SyncScope::write_mesh(uint32_t i) {
    return m_world.m_meshes.write(i);
}

Slot<LightData>::WriteGuard SyncScope::write_light(uint32_t i) {
    return m_world.m_lights.write(i);
}

const Slot<ObjectData>& SyncScope::object(uint32_t i) const {
    return m_world.m_objects[i];
}

const Slot<MeshData>& SyncScope::mesh(uint32_t i) const {
    return m_world.m_meshes[i];
}

Slot<CameraData>::WriteGuard SyncScope::write_camera(uint32_t i) {
    return m_world.m_cameras.write(i);
}

const Slot<LightData>& SyncScope::light(uint32_t i) const {
    return m_world.m_lights[i];
}

const Slot<CameraData>& SyncScope::camera(uint32_t i) const {
    return m_world.m_cameras[i];
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

void SyncScope::set_prim_path(uint32_t slot_index, PrimSlot::Kind kind, pxr::SdfPath path) {
    switch (kind) {
        case PrimSlot::Kind::Object:
            m_world.m_objects.set_prim_path(slot_index, path);
            break;
        case PrimSlot::Kind::Light:
            m_world.m_lights.set_prim_path(slot_index, path);
            break;
        case PrimSlot::Kind::Camera:
            m_world.m_cameras.set_prim_path(slot_index, path);
            break;
    }
    m_world.m_prim_slots[std::move(path)] = PrimSlot{kind, slot_index};
}

// --- RenderWorld accessors ---

boost::span<const Slot<ObjectData>> RenderWorld::get_objects() const {
    return m_objects.span();
}

boost::span<const Slot<MeshData>> RenderWorld::get_meshes() const {
    return m_meshes.span();
}

boost::span<const Slot<LightData>> RenderWorld::get_lights() const {
    return m_lights.span();
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

// --- RenderWorld read-only + clear ---

int RenderWorld::find_object_by_prim(const pxr::SdfPath& path) const {
    auto it = m_prim_slots.find(path);
    if (it == m_prim_slots.end() || it->second.kind != PrimSlot::Kind::Object) return -1;
    return static_cast<int>(it->second.index);
}

int RenderWorld::find_light_by_prim(const pxr::SdfPath& path) const {
    auto it = m_prim_slots.find(path);
    if (it == m_prim_slots.end() || it->second.kind != PrimSlot::Kind::Light) return -1;
    return static_cast<int>(it->second.index);
}

boost::span<const Slot<CameraData>> RenderWorld::get_cameras() const {
    return m_cameras.span();
}

int RenderWorld::find_camera_by_prim(const pxr::SdfPath& path) const {
    auto it = m_prim_slots.find(path);
    if (it == m_prim_slots.end() || it->second.kind != PrimSlot::Kind::Camera) return -1;
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
    auto lights = get_lights();

    if (m_light_version != m_cached_light_version) {
        // Structural change — full rebuild
        std::vector<Light> gpu_lights;
        for (const auto& slot : lights) {
            if (!slot.active()) continue;
            gpu_lights.push_back(to_light(slot.data()));
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

        // Snapshot all generations
        m_cached_light_generations.resize(lights.size());
        for (uint32_t i = 0; i < static_cast<uint32_t>(lights.size()); ++i) {
            m_cached_light_generations[i] = lights[i].generation();
        }
    } else {
        // Partial update: compare per-slot generation vs cached
        uint32_t gpu_idx = 0;
        for (uint32_t i = 0; i < static_cast<uint32_t>(lights.size()); ++i) {
            if (!lights[i].active()) continue;
            if (i < static_cast<uint32_t>(m_cached_light_generations.size()) &&
                lights[i].generation() != m_cached_light_generations[i]) {
                auto gl = to_light(lights[i].data());
                wgpuQueueWriteBuffer(queue, m_gpu_light_buffer.handle(), gpu_idx * sizeof(Light),
                                     &gl, sizeof(Light));
                m_cached_light_generations[i] = lights[i].generation();
            }
            ++gpu_idx;
        }
    }
}

void RenderWorld::upload_all_meshes(const webgpu::Device& device) {
    for (uint32_t i = 0; i < m_meshes.size(); ++i) {
        const auto& mesh = m_meshes[i].data();
        if (mesh.cpu_vertices.empty()) continue;

        PRECONDITION(!mesh.cpu_indices.empty());

        auto w = m_meshes.write(i);
        w->vertex_buffer = device.create_buffer(
            mesh.cpu_vertices.size() * sizeof(Vertex),
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst));
        wgpuQueueWriteBuffer(device.queue(), w->vertex_buffer.handle(), 0, mesh.cpu_vertices.data(),
                             mesh.cpu_vertices.size() * sizeof(Vertex));

        w->index_buffer = device.create_buffer(
            mesh.cpu_indices.size() * sizeof(uint32_t),
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst));
        wgpuQueueWriteBuffer(device.queue(), w->index_buffer.handle(), 0, mesh.cpu_indices.data(),
                             mesh.cpu_indices.size() * sizeof(uint32_t));

        w->index_count = static_cast<uint32_t>(mesh.cpu_indices.size());
    }
}

void RenderWorld::clear() {
    m_meshes.clear();
    m_objects.clear();
    m_materials.clear();
    m_lights.clear();
    m_cameras.clear();
    m_material_cache.clear();
    m_prim_slots.clear();
    m_gpu_light_buffer = {};
    m_gpu_material_buffer = {};
    m_gpu_light_count = 0;
    m_cached_light_version = UINT32_MAX;
    m_cached_material_version = UINT32_MAX;
    m_cached_light_generations.clear();
}

// --- update_transforms ---

void RenderWorld::update_transforms(const pxr::UsdStageRefPtr& stage,
                                    const std::vector<pxr::SdfPath>& dirty_paths) {
    for (const auto& dirty_path : dirty_paths) {
        for (const auto& [path, slot] : m_prim_slots) {
            if (!path.HasPrefix(dirty_path)) continue;

            auto prim = stage->GetPrimAtPath(path);
            if (!prim.IsValid()) continue;

            auto xf = compute_world_transform(prim);

            if (slot.kind == PrimSlot::Kind::Object) {
                auto w = m_objects.write(slot.index);
                w->transform = xf;
            } else {
                auto w = m_lights.write(slot.index);
                w->transform = xf;
                if (w->type == LightData::Type::Distant) {
                    glm::vec4 local_dir(0.0f, 0.0f, -1.0f, 0.0f);
                    w->direction = glm::normalize(glm::vec3(xf * local_dir));
                }
                ++m_light_version;
            }
        }
    }
}

}  // namespace pts::rendering
