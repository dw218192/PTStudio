#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/adapterHelpers.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/webgpu/device.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
#include <stb_image.h>

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

    int w = 0, h = 0, channels = 0;
    auto* data = stbi_load(resolved_path.c_str(), &w, &h, &channels, 4);
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

void RenderWorld::prepare_gpu_buffers(const webgpu::Device& device, WGPUQueue queue) {
    PTS_ZONE_SCOPED;
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

    // --- Scene BVH ---
    if (m_mesh_version != m_cached_bvh_mesh_version) {
        PTS_ZONE_NAMED("BVH rebuild");
        auto objects = get_objects();
        auto meshes_span = get_meshes();

        // Flatten all world-space triangle AABBs
        std::vector<AABB> tri_aabbs;
        for (const auto& obj : objects) {
            if (!obj.active()) continue;
            const auto& mesh = meshes_span[obj->mesh_index];
            if (!mesh.active() || mesh->cpu_vertices.empty() || mesh->cpu_indices.empty()) continue;
            const auto& xform = obj->transform;
            for (uint32_t i = 0; i + 2 < static_cast<uint32_t>(mesh->cpu_indices.size()); i += 3) {
                AABB a;
                for (int vi = 0; vi < 3; ++vi) {
                    const auto& v = mesh->cpu_vertices[mesh->cpu_indices[i + vi]];
                    a.expand(glm::vec3(
                        xform * glm::vec4(v.position[0], v.position[1], v.position[2], 1.0f)));
                }
                tri_aabbs.push_back(a);
            }
        }

        m_scene_bvh.build(tri_aabbs, static_cast<uint32_t>(tri_aabbs.size()));
        m_scene_bvh.upload(device, queue);
        m_cached_bvh_mesh_version = m_mesh_version;
    }

    // --- Texture array ---
    if (m_texture_version != m_cached_texture_version) {
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
            m_texture_images.empty() ? 1 : static_cast<uint32_t>(m_texture_images.size());
        uint32_t tex_w = m_texture_images.empty() ? 1 : m_texture_size;
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

        if (m_texture_images.empty()) {
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
            for (uint32_t i = 0; i < static_cast<uint32_t>(m_texture_images.size()); ++i) {
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
                wgpuQueueWriteTexture(queue, &dst, m_texture_images[i].pixels.data(),
                                      m_texture_images[i].pixels.size(), &layout, &extent);
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

        m_cached_texture_version = m_texture_version;
    }
}

const BVH& RenderWorld::scene_bvh() const {
    return m_scene_bvh;
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

// --- Shadow data ---

namespace {
constexpr std::size_t k_min_shadow_info_size = sizeof(ShadowInfo);  // 80 bytes
}  // namespace

void RenderWorld::set_shadow_data(boost::span<const ShadowInfo> infos, const webgpu::Device& device,
                                  WGPUQueue queue) {
    auto info_bytes = std::max(k_min_shadow_info_size,
                               static_cast<std::size_t>(infos.size()) * sizeof(ShadowInfo));
    if (!m_shadow_info_buffer.is_valid() || m_shadow_info_buffer.size() < info_bytes) {
        m_shadow_info_buffer = device.create_buffer(
            info_bytes,
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
    }
    if (!infos.empty()) {
        wgpuQueueWriteBuffer(queue, m_shadow_info_buffer.handle(), 0, infos.data(),
                             infos.size() * sizeof(ShadowInfo));
    }

    // Count entries with has_shadow set
    uint32_t active = 0;
    for (const auto& si : infos) {
        if (si.has_shadow) ++active;
    }
    m_shadow_count = active;
}

void RenderWorld::clear_shadow_data() {
    m_shadow_count = 0;
}

const webgpu::Buffer& RenderWorld::shadow_info_buffer() const {
    return m_shadow_info_buffer;
}

uint32_t RenderWorld::shadow_count() const {
    return m_shadow_count;
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
    clear_shadow_data();

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
                    ++m_mesh_version;
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

}  // namespace pts::rendering
