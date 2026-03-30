#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/adapterHelpers.h>
#include <core/rendering/preparedSceneData.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/webgpu/device.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
#include <spdlog/spdlog.h>
#include <stb_image.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <limits>

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
    // Skip the reserved default material at index 0.
    return {m_materials.data() + 1, m_materials.size() - 1};
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

WGPUTextureView RenderWorld::texture_array_view() const {
    return m_texture_array_view;
}

WGPUSampler RenderWorld::texture_sampler() const {
    return m_texture_sampler;
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

// --- Texture loading ---

namespace {
void resize_rgba8(const uint8_t* src, uint32_t src_w, uint32_t src_h, uint8_t* dst, uint32_t dst_w,
                  uint32_t dst_h) {
    for (uint32_t y = 0; y < dst_h; ++y) {
        float v = static_cast<float>(y) * static_cast<float>(src_h) / static_cast<float>(dst_h);
        auto y0 = static_cast<uint32_t>(v);
        float fy = v - static_cast<float>(y0);
        uint32_t y1 = std::min(y0 + 1, src_h - 1);
        for (uint32_t x = 0; x < dst_w; ++x) {
            float u = static_cast<float>(x) * static_cast<float>(src_w) / static_cast<float>(dst_w);
            auto x0 = static_cast<uint32_t>(u);
            float fx = u - static_cast<float>(x0);
            uint32_t x1 = std::min(x0 + 1, src_w - 1);
            for (int c = 0; c < 4; ++c) {
                float p00 = src[(y0 * src_w + x0) * 4 + c];
                float p10 = src[(y0 * src_w + x1) * 4 + c];
                float p01 = src[(y1 * src_w + x0) * 4 + c];
                float p11 = src[(y1 * src_w + x1) * 4 + c];
                float val = p00 * (1 - fx) * (1 - fy) + p10 * fx * (1 - fy) + p01 * (1 - fx) * fy +
                            p11 * fx * fy;
                dst[(y * dst_w + x) * 4 + c] =
                    static_cast<uint8_t>(std::min(std::max(val, 0.0f), 255.0f));
            }
        }
    }
}
}  // namespace

uint32_t SyncScope::load_texture(const std::string& resolved_path) {
    auto it = m_world.m_texture_cache.find(resolved_path);
    if (it != m_world.m_texture_cache.end()) return it->second;

    auto asset = pxr::ArGetResolver().OpenAsset(pxr::ArResolvedPath(resolved_path));
    if (!asset) return UINT32_MAX;

    auto buffer = asset->GetBuffer();
    auto size = asset->GetSize();
    CHECK_MSG(buffer != nullptr, "ArAsset::GetBuffer() returned null for opened asset");

    int w = 0, h = 0, channels = 0;
    auto* data = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(buffer.get()),
                                       static_cast<int>(size), &w, &h, &channels, 4);
    if (!data) return UINT32_MAX;

    auto index = static_cast<uint32_t>(m_world.m_texture_images.size());
    auto tex_size = m_world.m_texture_size;

    RenderWorld::ImageData img;
    if (static_cast<uint32_t>(w) == tex_size && static_cast<uint32_t>(h) == tex_size) {
        img.pixels.assign(data, data + tex_size * tex_size * 4);
    } else {
        img.pixels.resize(tex_size * tex_size * 4);
        resize_rgba8(data, static_cast<uint32_t>(w), static_cast<uint32_t>(h), img.pixels.data(),
                     tex_size, tex_size);
    }
    stbi_image_free(data);

    img.width = tex_size;
    img.height = tex_size;
    m_world.m_texture_images.push_back(std::move(img));
    m_world.m_texture_cache[resolved_path] = index;
    ++m_world.m_texture_version;
    return index;
}

// --- GPU buffer upload ---

namespace {
constexpr std::size_t k_min_material_buffer_size = sizeof(Material);
constexpr std::size_t k_min_light_buffer_size = sizeof(Light);  // 48 bytes
}  // namespace

PreparedSceneData RenderWorld::prepare_scene_data() {
    PTS_ZONE_SCOPED;
    PreparedSceneData data;

    // --- Materials ---
    if (m_material_version != m_cached_material_version) {
        data.materials = m_materials;
        data.materials_dirty = true;
        m_cached_material_version = m_material_version;
    }

    // --- Lights ---
    auto lights = get_lights();

    if (m_light_version != m_cached_light_version) {
        // Structural change — full rebuild
        for (const auto& slot : lights) {
            if (!slot.active()) continue;
            data.gpu_lights.push_back(to_light(slot.data()));
        }

        // Default fallback: single distant light when scene has no lights
        if (data.gpu_lights.empty()) {
            Light def{};
            def.type = 0;
            def.direction_or_pos = glm::normalize(glm::vec3(0.3f, -1.0f, 0.5f));
            def.color = {1.0f, 0.95f, 0.9f};
            def.intensity = 1.0f;
            data.gpu_lights.push_back(def);
        }

        data.lights_dirty = true;
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
                data.partial_light_updates.push_back({gpu_idx, to_light(lights[i].data())});
                m_cached_light_generations[i] = lights[i].generation();
            }
            ++gpu_idx;
        }
    }

    // --- Two-level BVH (BLAS per mesh, TLAS over instances) ---
    {
        auto objects = get_objects();
        auto meshes_span = get_meshes();

        // Step 1: Collect dirty meshes and pre-populate BLAS cache entries (serial)
        PTS_ZONE_NAMED("BLAS build");
        std::vector<uint32_t> dirty_meshes;
        for (const auto& obj : objects) {
            if (!obj.active()) continue;
            uint32_t mesh_idx = obj->mesh_index;
            const auto& mesh = meshes_span[mesh_idx];
            if (!mesh.active() || mesh->cpu_vertices.empty() || mesh->cpu_indices.empty()) continue;

            // Pre-populate cache entry (must happen before parallel_for)
            auto& blas = m_blas_cache[mesh_idx];
            if (blas.generation == mesh.generation()) continue;
            if (std::find(dirty_meshes.begin(), dirty_meshes.end(), mesh_idx) ==
                dirty_meshes.end()) {
                dirty_meshes.push_back(mesh_idx);
            }
        }

        // Build BLAS in parallel (each mesh is independent)
        tbb::parallel_for(tbb::blocked_range<size_t>(0, dirty_meshes.size()),
                          [&](const tbb::blocked_range<size_t>& range) {
                              for (size_t i = range.begin(); i < range.end(); ++i) {
                                  uint32_t mesh_idx = dirty_meshes[i];
                                  auto& blas = m_blas_cache[mesh_idx];
                                  const auto& mesh = meshes_span[mesh_idx];
                                  blas.tris = blas.bvh.build_from_mesh(mesh->cpu_vertices,
                                                                       mesh->cpu_indices);
                                  blas.generation = mesh.generation();
                              }
                          });
        bool any_blas_dirty = !dirty_meshes.empty();

        // Step 2: Build instance array + TLAS
        bool need_rebuild = any_blas_dirty || m_transform_version != m_cached_transform_version ||
                            m_mesh_version != m_cached_geometry_version;

        if (need_rebuild) {
            // Collect instances and their world-space AABBs
            struct InstanceInfo {
                uint32_t mesh_index;
                uint32_t material_index;
                glm::mat4 transform;
            };
            std::vector<InstanceInfo> instances;
            std::vector<AABB> world_aabbs;

            for (const auto& obj : objects) {
                if (!obj.active()) continue;
                uint32_t mesh_idx = obj->mesh_index;
                const auto& mesh = meshes_span[mesh_idx];
                if (!mesh.active() || mesh->cpu_vertices.empty() || mesh->cpu_indices.empty())
                    continue;

                INVARIANT(m_blas_cache.count(mesh_idx) > 0);

                AABB local_aabb = AABB::from_min_max(mesh->local_aabb_min, mesh->local_aabb_max);
                world_aabbs.push_back(transform_aabb(local_aabb, obj->transform));
                instances.push_back({mesh_idx, obj->material_index, obj->transform});
            }

            auto inst_count = static_cast<uint32_t>(instances.size());

            // Build TLAS from world-space AABBs
            {
                PTS_ZONE_NAMED("TLAS build");
                m_tlas.build(world_aabbs, inst_count);
            }
            uint32_t tlas_nc = m_tlas.node_count();

            // Build per-mesh offset table (unique meshes only)
            struct MeshOffset {
                uint32_t blas_offset;
                uint32_t tri_offset;
            };
            std::unordered_map<uint32_t, MeshOffset> mesh_offsets;
            std::vector<uint32_t> unique_meshes;
            uint32_t running_blas_offset = 0;
            uint32_t running_tri_offset = 0;

            for (const auto& inst : instances) {
                if (mesh_offsets.count(inst.mesh_index) > 0) continue;
                unique_meshes.push_back(inst.mesh_index);
                const auto& blas = m_blas_cache[inst.mesh_index];
                mesh_offsets[inst.mesh_index] = {tlas_nc + running_blas_offset, running_tri_offset};
                running_blas_offset += blas.bvh.node_count();
                running_tri_offset += static_cast<uint32_t>(blas.tris.size());
            }

            // Build GPUInstance array
            std::vector<GPUInstance> gpu_instances(inst_count);
            for (uint32_t i = 0; i < inst_count; ++i) {
                const auto& inst = instances[i];
                const auto& offset = mesh_offsets[inst.mesh_index];
                const auto& blas = m_blas_cache[inst.mesh_index];

                gpu_instances[i].transform = inst.transform;
                gpu_instances[i].inv_transform = glm::inverse(inst.transform);
                gpu_instances[i].blas_offset = offset.blas_offset;
                gpu_instances[i].tri_offset = offset.tri_offset;
                gpu_instances[i].tri_count = static_cast<uint32_t>(blas.tris.size());
                gpu_instances[i].material_index = inst.material_index;
            }

            // Reorder instances by TLAS tri_indices
            if (!m_tlas.tri_indices().empty() && inst_count > 0) {
                INVARIANT(m_tlas.tri_indices().size() == inst_count);
                std::vector<GPUInstance> reordered(inst_count);
                for (uint32_t i = 0; i < inst_count; ++i) {
                    reordered[i] = gpu_instances[m_tlas.tri_indices()[i]];
                }
                gpu_instances = std::move(reordered);
            }

            // Concatenate TLAS + BLAS nodes
            std::vector<BlasEntry> blas_entries;
            blas_entries.reserve(unique_meshes.size());
            for (uint32_t mi : unique_meshes) {
                blas_entries.push_back({&m_blas_cache[mi].bvh, mesh_offsets[mi].tri_offset});
            }
            data.all_nodes = m_tlas.concatenate_nodes(blas_entries);

            // Concatenate triangles
            data.all_tris.reserve(running_tri_offset);
            for (uint32_t mesh_idx : unique_meshes) {
                const auto& blas = m_blas_cache[mesh_idx];
                data.all_tris.insert(data.all_tris.end(), blas.tris.begin(), blas.tris.end());
            }

            data.gpu_instances = std::move(gpu_instances);
            data.tlas_node_count = tlas_nc;
            data.instance_count = inst_count;
            data.geometry_dirty = true;

            m_cached_transform_version = m_transform_version;
            m_cached_geometry_version = m_mesh_version;
        }
    }

    // --- Texture array ---
    if (m_texture_version != m_cached_texture_version) {
        data.texture_size = m_texture_size;
        for (const auto& img : m_texture_images) {
            data.texture_layers.push_back({img.pixels.data(), img.width, img.height});
        }
        data.textures_dirty = true;
        m_cached_texture_version = m_texture_version;
    }

    return data;
}

void RenderWorld::upload_prepared_data(const webgpu::Device& device, WGPUQueue queue,
                                       const PreparedSceneData& data) {
    PTS_ZONE_SCOPED;

    // --- Materials ---
    if (data.materials_dirty) {
        auto material_count = static_cast<uint32_t>(data.materials.size());
        auto required_size = std::max(k_min_material_buffer_size,
                                      static_cast<std::size_t>(material_count) * sizeof(Material));

        if (required_size > m_gpu_material_buffer.size()) {
            m_gpu_material_buffer = device.create_buffer(
                required_size,
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
        }

        if (material_count > 0) {
            wgpuQueueWriteBuffer(queue, m_gpu_material_buffer.handle(), 0, data.materials.data(),
                                 material_count * sizeof(Material));
        }
    }

    // --- Lights ---
    if (data.lights_dirty) {
        auto buf_size = std::max(k_min_light_buffer_size, data.gpu_lights.size() * sizeof(Light));

        if (!m_gpu_light_buffer.is_valid() || m_gpu_light_buffer.size() < buf_size) {
            m_gpu_light_buffer = device.create_buffer(
                buf_size,
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
        }

        wgpuQueueWriteBuffer(queue, m_gpu_light_buffer.handle(), 0, data.gpu_lights.data(),
                             data.gpu_lights.size() * sizeof(Light));
        m_gpu_light_count = static_cast<uint32_t>(data.gpu_lights.size());
    } else {
        for (const auto& update : data.partial_light_updates) {
            wgpuQueueWriteBuffer(queue, m_gpu_light_buffer.handle(),
                                 update.gpu_index * sizeof(Light), &update.data, sizeof(Light));
        }
    }

    // --- BVH + geometry ---
    if (data.geometry_dirty) {
        // Upload concatenated TLAS + BLAS nodes
        auto node_bytes = std::max(sizeof(BVHNode), data.all_nodes.size() * sizeof(BVHNode));
        if (!m_gpu_bvh_nodes.is_valid() || m_gpu_bvh_nodes.size() < node_bytes) {
            m_gpu_bvh_nodes = device.create_buffer(
                node_bytes,
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
        }
        if (!data.all_nodes.empty()) {
            wgpuQueueWriteBuffer(queue, m_gpu_bvh_nodes.handle(), 0, data.all_nodes.data(),
                                 data.all_nodes.size() * sizeof(BVHNode));
        }

        // Upload concatenated triangles
        auto tri_bytes =
            std::max(sizeof(PackedTriangle), data.all_tris.size() * sizeof(PackedTriangle));
        if (!m_gpu_triangles.is_valid() || m_gpu_triangles.size() < tri_bytes) {
            m_gpu_triangles = device.create_buffer(
                tri_bytes,
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
        }
        if (!data.all_tris.empty()) {
            wgpuQueueWriteBuffer(queue, m_gpu_triangles.handle(), 0, data.all_tris.data(),
                                 data.all_tris.size() * sizeof(PackedTriangle));
        }

        // Upload instances
        auto inst_bytes =
            std::max(sizeof(GPUInstance), data.gpu_instances.size() * sizeof(GPUInstance));
        if (!m_gpu_instances.is_valid() || m_gpu_instances.size() < inst_bytes) {
            m_gpu_instances = device.create_buffer(
                inst_bytes,
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
        }
        if (!data.gpu_instances.empty()) {
            wgpuQueueWriteBuffer(queue, m_gpu_instances.handle(), 0, data.gpu_instances.data(),
                                 data.gpu_instances.size() * sizeof(GPUInstance));
        }

        m_tlas_node_count = data.tlas_node_count;
        m_instance_count = data.instance_count;
    }

    // --- Texture array ---
    if (data.textures_dirty) {
        PTS_ZONE_NAMED("texture array upload");
        // Release old resources
        if (m_texture_array_view) {
            wgpuTextureViewRelease(m_texture_array_view);
            m_texture_array_view = nullptr;
        }
        if (m_texture_array) {
            wgpuTextureDestroy(m_texture_array);
            wgpuTextureRelease(m_texture_array);
            m_texture_array = nullptr;
        }
        if (m_texture_sampler) {
            wgpuSamplerRelease(m_texture_sampler);
            m_texture_sampler = nullptr;
        }

        uint32_t layer_count =
            data.texture_layers.empty() ? 1 : static_cast<uint32_t>(data.texture_layers.size());
        uint32_t tex_w = data.texture_layers.empty() ? 1 : data.texture_size;
        uint32_t tex_h = tex_w;

        WGPUTextureDescriptor tex_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        tex_desc.usage = static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding |
                                                       WGPUTextureUsage_CopyDst);
        tex_desc.dimension = WGPUTextureDimension_2D;
        tex_desc.size = {tex_w, tex_h, layer_count};
        tex_desc.format = WGPUTextureFormat_RGBA8Unorm;
        tex_desc.mipLevelCount = 1;
        m_texture_array = wgpuDeviceCreateTexture(device.handle(), &tex_desc);
        POSTCONDITION(m_texture_array);

        if (data.texture_layers.empty()) {
            // 1x1 white placeholder
            uint8_t white[] = {255, 255, 255, 255};
            WGPUTexelCopyTextureInfo dst = {};
            dst.texture = m_texture_array;
            dst.mipLevel = 0;
            dst.origin = {0, 0, 0};
            dst.aspect = WGPUTextureAspect_All;
            WGPUTexelCopyBufferLayout layout = {};
            layout.offset = 0;
            layout.bytesPerRow = 4;
            layout.rowsPerImage = 1;
            WGPUExtent3D extent = {1, 1, 1};
            wgpuQueueWriteTexture(queue, &dst, white, sizeof(white), &layout, &extent);
        } else {
            uint32_t bytes_per_row = tex_w * 4;
            for (uint32_t i = 0; i < static_cast<uint32_t>(data.texture_layers.size()); ++i) {
                const auto& layer = data.texture_layers[i];
                WGPUTexelCopyTextureInfo dst = {};
                dst.texture = m_texture_array;
                dst.mipLevel = 0;
                dst.origin = {0, 0, i};
                dst.aspect = WGPUTextureAspect_All;
                WGPUTexelCopyBufferLayout layout = {};
                layout.offset = 0;
                layout.bytesPerRow = bytes_per_row;
                layout.rowsPerImage = tex_h;
                WGPUExtent3D extent = {tex_w, tex_h, 1};
                wgpuQueueWriteTexture(queue, &dst, layer.pixels,
                                      static_cast<std::size_t>(tex_w) * tex_h * 4, &layout,
                                      &extent);
            }
        }

        WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
        view_desc.format = WGPUTextureFormat_RGBA8Unorm;
        view_desc.dimension = WGPUTextureViewDimension_2DArray;
        view_desc.baseMipLevel = 0;
        view_desc.mipLevelCount = 1;
        view_desc.baseArrayLayer = 0;
        view_desc.arrayLayerCount = layer_count;
        m_texture_array_view = wgpuTextureCreateView(m_texture_array, &view_desc);
        POSTCONDITION(m_texture_array_view);

        WGPUSamplerDescriptor sampler_desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
        sampler_desc.addressModeU = WGPUAddressMode_Repeat;
        sampler_desc.addressModeV = WGPUAddressMode_Repeat;
        sampler_desc.addressModeW = WGPUAddressMode_Repeat;
        sampler_desc.magFilter = WGPUFilterMode_Linear;
        sampler_desc.minFilter = WGPUFilterMode_Linear;
        sampler_desc.mipmapFilter = WGPUMipmapFilterMode_Linear;
        sampler_desc.maxAnisotropy = 1;
        m_texture_sampler = wgpuDeviceCreateSampler(device.handle(), &sampler_desc);
        POSTCONDITION(m_texture_sampler);
    }
}

void RenderWorld::prepare_gpu_buffers(const webgpu::Device& device, WGPUQueue queue) {
    auto prepared = prepare_scene_data();
    upload_prepared_data(device, queue, prepared);
}

AABB RenderWorld::scene_bounds() const {
    return m_tlas.scene_bounds();
}

const webgpu::Buffer& RenderWorld::bvh_node_buffer() const {
    return m_gpu_bvh_nodes;
}

const webgpu::Buffer& RenderWorld::triangle_buffer() const {
    return m_gpu_triangles;
}

const webgpu::Buffer& RenderWorld::instance_buffer() const {
    return m_gpu_instances;
}

uint32_t RenderWorld::tlas_node_count() const {
    return m_tlas_node_count;
}

uint32_t RenderWorld::instance_count() const {
    return m_instance_count;
}

void RenderWorld::upload_all_meshes(const webgpu::Device& device) {
    PTS_ZONE_SCOPED;
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

        // Position-only buffer for picking and depth prepass, plus local AABB
        auto vert_count = mesh.cpu_vertices.size();
        std::vector<glm::vec3> positions(vert_count);
        glm::vec3 aabb_min(std::numeric_limits<float>::max());
        glm::vec3 aabb_max(std::numeric_limits<float>::lowest());
        for (size_t v = 0; v < vert_count; ++v) {
            positions[v] = glm::make_vec3(mesh.cpu_vertices[v].position);
            aabb_min = glm::min(aabb_min, positions[v]);
            aabb_max = glm::max(aabb_max, positions[v]);
        }
        w->local_aabb_min = aabb_min;
        w->local_aabb_max = aabb_max;
        w->position_buffer = device.create_buffer(
            vert_count * sizeof(glm::vec3),
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst));
        wgpuQueueWriteBuffer(device.queue(), w->position_buffer.handle(), 0, positions.data(),
                             vert_count * sizeof(glm::vec3));
    }
}

void RenderWorld::clear() {
    m_meshes.clear();
    m_objects.clear();
    m_materials.clear();
    m_materials.push_back(Material{});  // default material at index 0
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

    // Two-level BVH state
    m_blas_cache.clear();
    m_tlas = {};
    m_gpu_bvh_nodes = {};
    m_gpu_triangles = {};
    m_gpu_instances = {};
    m_tlas_node_count = 0;
    m_instance_count = 0;
    m_cached_transform_version = UINT32_MAX;
    m_cached_geometry_version = UINT32_MAX;

    // Texture state
    m_texture_images.clear();
    m_texture_cache.clear();
    if (m_texture_array_view) {
        wgpuTextureViewRelease(m_texture_array_view);
        m_texture_array_view = nullptr;
    }
    if (m_texture_array) {
        wgpuTextureDestroy(m_texture_array);
        wgpuTextureRelease(m_texture_array);
        m_texture_array = nullptr;
    }
    if (m_texture_sampler) {
        wgpuSamplerRelease(m_texture_sampler);
        m_texture_sampler = nullptr;
    }
    m_texture_version = 0;
    m_cached_texture_version = UINT32_MAX;

    // IBL state
    m_ibl = {};
    m_ibl_env_path.clear();
    m_ibl_light_version = UINT32_MAX;
    m_ibl_uniform_color = glm::vec3(-1.0f);
    m_ibl_up_axis = UpAxis::Y;
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

            switch (slot.kind) {
                case PrimSlot::Kind::Object: {
                    auto w = m_objects.write(slot.index);
                    w->transform = xf;
                    ++m_transform_version;
                    break;
                }
                case PrimSlot::Kind::Light: {
                    auto w = m_lights.write(slot.index);
                    w->transform = xf;
                    if (w->type == LightData::Type::Distant) {
                        glm::vec4 local_dir(0.0f, 0.0f, -1.0f, 0.0f);
                        w->direction = glm::normalize(glm::vec3(xf * local_dir));
                    }
                    ++m_light_version;
                    break;
                }
                case PrimSlot::Kind::Camera: {
                    auto w = m_cameras.write(slot.index);
                    w->view_matrix = glm::inverse(xf);
                    break;
                }
            }
        }
    }
}

// --- IBL ---

IblResources& RenderWorld::ibl_resources() {
    return m_ibl;
}

const IblResources& RenderWorld::ibl_resources() const {
    return m_ibl;
}

const IblPipelines& RenderWorld::ibl_pipelines() const {
    PRECONDITION(m_ibl_pipelines);
    return *m_ibl_pipelines;
}

void RenderWorld::update_ibl(const webgpu::Device& device, WGPUQueue queue, UpAxis up_axis) {
    PTS_ZONE_SCOPED;

    // Lazy-init pipelines on first call
    if (!m_ibl_pipelines) {
        m_ibl_pipelines = std::make_unique<IblPipelines>();
        m_ibl_pipelines->init(device, queue);
    }

    // Only re-evaluate when lights change
    if (m_ibl_light_version == m_light_version) return;

    // Find first dome light
    const LightData* dome = nullptr;
    auto lights = get_lights();
    for (const auto& slot : lights) {
        if (!slot.active()) continue;
        if (slot.data().type == LightData::Type::Dome) {
            dome = &slot.data();
            break;
        }
    }

    if (!dome) {
        // No dome light — black ambient
        if (m_ibl_env_path.empty() && m_ibl_uniform_color == glm::vec3(0.0f)) return;
        m_ibl.set_uniform_environment(device, queue, 0.0f, 0.0f, 0.0f);
        m_ibl_env_path.clear();
        m_ibl_uniform_color = glm::vec3(0.0f);
        m_ibl_light_version = m_light_version;
        return;
    }

    if (!dome->env_texture_path.empty()) {
        // HDR environment map
        if (dome->env_texture_path == m_ibl_env_path && up_axis == m_ibl_up_axis) return;

        auto asset = pxr::ArGetResolver().OpenAsset(pxr::ArResolvedPath(dome->env_texture_path));
        if (!asset) {
            spdlog::warn("Failed to open HDR environment: {}", dome->env_texture_path);
            return;
        }

        auto buffer = asset->GetBuffer();
        auto size = asset->GetSize();
        if (!buffer) {
            spdlog::warn("Empty HDR environment asset: {}", dome->env_texture_path);
            return;
        }

        int w = 0, h = 0, channels = 0;
        float* data = stbi_loadf_from_memory(reinterpret_cast<const stbi_uc*>(buffer.get()),
                                             static_cast<int>(size), &w, &h, &channels, 4);
        if (!data) {
            spdlog::warn("Failed to decode HDR environment: {}", dome->env_texture_path);
            return;
        }

        m_ibl.set_environment(*m_ibl_pipelines, device, queue, data, static_cast<uint32_t>(w),
                              static_cast<uint32_t>(h), up_axis);
        stbi_image_free(data);

        m_ibl_env_path = dome->env_texture_path;
        m_ibl_up_axis = up_axis;
        m_ibl_uniform_color = glm::vec3(-1.0f);  // invalidate uniform sentinel
    } else {
        // Uniform color environment: dome color * intensity
        glm::vec3 c = dome->color * dome->intensity;
        if (m_ibl_env_path.empty() && m_ibl_uniform_color == c) return;

        m_ibl.set_uniform_environment(device, queue, c.r, c.g, c.b);
        m_ibl_env_path.clear();
        m_ibl_uniform_color = c;
    }

    m_ibl_light_version = m_light_version;
}

}  // namespace pts::rendering
