#include <Iex.h>
#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfIO.h>
#include <ImfInputFile.h>
#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/rendering/adapterHelpers.h>
#include <core/rendering/halfFloat.h>
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
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <limits>

namespace pts::rendering {

namespace {

// Field-mask shorthands for the registered consumers. Lifecycle is the
// "structural change" bit -- SlotMap auto-sets it on insert/erase.
//
// LightField subscriptions:
//   light buffer: every field that affects to_light() output, plus Lifecycle
//                 (so insert/erase trigger a full rebuild + buffer resize).
constexpr LightField k_light_buffer_fields =
    LightField::Lifecycle | LightField::Transform | LightField::Color | LightField::Intensity |
    LightField::Type | LightField::Direction | LightField::Geometry | LightField::MaterialIndex;

//   tlas (lights): proxy-mesh emitter geometry -- only transform/visibility/
//                  mesh_index/material_index affect the TLAS instance set.
constexpr LightField k_light_tlas_fields = LightField::Lifecycle | LightField::Transform |
                                           LightField::Visibility | LightField::MeshIndex |
                                           LightField::MaterialIndex;

//   ibl: dome-light state (type, color, intensity, env texture).
constexpr LightField k_light_ibl_fields = LightField::Lifecycle | LightField::Type |
                                          LightField::Color | LightField::Intensity |
                                          LightField::EnvTexture;

constexpr ObjectField k_object_tlas_fields = ObjectField::Lifecycle | ObjectField::Transform |
                                             ObjectField::Visibility | ObjectField::MeshIndex |
                                             ObjectField::MaterialIndex;

constexpr MeshField k_mesh_blas_fields = MeshField::Lifecycle | MeshField::Geometry;

}  // namespace

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

// --- RenderWorld lifecycle ---

RenderWorld::RenderWorld() {
    m_materials.push_back(Material{});
    register_internal_consumers();
}

void RenderWorld::register_internal_consumers() {
    m_lights_buffer_consumer = m_lights.register_consumer(k_light_buffer_fields);
    m_lights_tlas_consumer = m_lights.register_consumer(k_light_tlas_fields);
    m_lights_ibl_consumer = m_lights.register_consumer(k_light_ibl_fields);
    m_objects_tlas_consumer = m_objects.register_consumer(k_object_tlas_fields);
    m_meshes_blas_consumer = m_meshes.register_consumer(k_mesh_blas_fields);
    m_materials_consumer = register_world_consumer(WorldDirty::Materials);
    m_textures_consumer = register_world_consumer(WorldDirty::SceneTextures);
}

// --- WorldDirty consumer machinery ---

RenderWorld::WorldConsumerId RenderWorld::register_world_consumer(WorldDirty subscription) {
    WorldConsumerState state;
    state.subscription = subscription;
    // New consumers see existing state as fully dirty -- their first drain
    // will produce a complete rebuild.
    state.pending = subscription;
    m_world_consumers.push_back(state);
    return static_cast<WorldConsumerId>(m_world_consumers.size() - 1);
}

void RenderWorld::mark_world_dirty(WorldDirty bits) {
    for (auto& c : m_world_consumers) {
        c.pending |= (bits & c.subscription);
    }
}

bool RenderWorld::any_world_dirty(WorldConsumerId id, WorldDirty query) const {
    PRECONDITION(id < m_world_consumers.size());
    return static_cast<uint32_t>(m_world_consumers[id].pending & query) != 0;
}

WorldDirty RenderWorld::take_world_dirty(WorldConsumerId id, WorldDirty query) {
    PRECONDITION(id < m_world_consumers.size());
    auto& c = m_world_consumers[id];
    auto taken = c.pending & query;
    c.pending = c.pending & ~query;
    return taken;
}

// --- SyncScope ---

SyncScope::SyncScope(RenderWorld& world) : m_world(world) {
}

SyncScope::~SyncScope() {
    // Materials are stored in a flat vector and aren't per-slot tracked --
    // we conservatively flag them dirty after every scope (preserves the
    // pre-refactor "always rebuild materials" behavior). Per-slot SlotMap
    // dirty bits handle lights/objects/meshes; load_texture marks textures
    // explicitly.
    m_world.mark_world_dirty(WorldDirty::Materials);
}

SyncScope RenderWorld::begin_sync() {
    return SyncScope(*this);
}

// --- Slot allocation (via SyncScope) ---

uint32_t SyncScope::alloc_object(const pxr::SdfPath& path) {
    return m_world.m_objects.insert(path, ObjectData{}).index();
}

uint32_t SyncScope::alloc_mesh(const pxr::SdfPath& path) {
    return m_world.m_meshes.insert(path, MeshData{}).index();
}

uint32_t SyncScope::alloc_light(const pxr::SdfPath& path) {
    return m_world.m_lights.insert(path, LightData{}).index();
}

uint32_t SyncScope::alloc_camera(const pxr::SdfPath& path) {
    return m_world.m_cameras.insert(path, CameraData{}).index();
}

void SyncScope::free_object(const pxr::SdfPath& path) {
    m_world.m_objects.erase(path);
}

void SyncScope::free_mesh(const pxr::SdfPath& path) {
    m_world.m_meshes.erase(path);
}

void SyncScope::free_light(const pxr::SdfPath& path) {
    m_world.m_lights.erase(path);
}

void SyncScope::free_camera(const pxr::SdfPath& path) {
    m_world.m_cameras.erase(path);
}

// --- SyncScope accessors ---

const ObjectData& SyncScope::object(uint32_t i) const {
    return m_world.m_objects.at(i);
}

const MeshData& SyncScope::mesh(uint32_t i) const {
    return m_world.m_meshes.at(i);
}

const LightData& SyncScope::light(uint32_t i) const {
    return m_world.m_lights.at(i);
}

const CameraData& SyncScope::camera(uint32_t i) const {
    return m_world.m_cameras.at(i);
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

// --- RenderWorld accessors ---

const ObjectSlotMap& RenderWorld::get_objects() const {
    return m_objects;
}

const MeshSlotMap& RenderWorld::get_meshes() const {
    return m_meshes;
}

const LightSlotMap& RenderWorld::get_lights() const {
    return m_lights;
}

const CameraSlotMap& RenderWorld::get_cameras() const {
    return m_cameras;
}

boost::span<const Material> RenderWorld::get_materials() const {
    // Skip the reserved default material at index 0.
    return {m_materials.data() + 1, m_materials.size() - 1};
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
    auto h = m_objects.find(path);
    if (!h) return -1;
    return static_cast<int>(h.index());
}

int RenderWorld::find_light_by_prim(const pxr::SdfPath& path) const {
    auto h = m_lights.find(path);
    if (!h) return -1;
    return static_cast<int>(h.index());
}

int RenderWorld::find_camera_by_prim(const pxr::SdfPath& path) const {
    auto h = m_cameras.find(path);
    if (!h) return -1;
    return static_cast<int>(h.index());
}

// --- Texture loading ---

namespace {

std::string get_asset_extension(const std::string& path) {
    auto ext = pxr::ArGetResolver().GetExtension(path);
    for (auto& c : ext)
        if (c >= 'A' && c <= 'Z') c += 32;
    return ext;
}
bool has_exr_extension(const std::string& path) {
    return get_asset_extension(path) == "exr";
}
bool has_hdr_extension(const std::string& path) {
    return get_asset_extension(path) == "hdr";
}

class MemoryIStream : public Imf::IStream {
   public:
    MemoryIStream(const char* name, const unsigned char* data, size_t size)
        : Imf::IStream(name), m_data(data), m_size(size), m_pos(0) {
    }

    bool isMemoryMapped() const override {
        return true;
    }

    bool read(char c[], int n) override {
        if (m_pos + n > m_size) throw Iex::InputExc("Unexpected end of EXR data");
        std::memcpy(c, m_data + m_pos, n);
        m_pos += n;
        return m_pos < m_size;
    }

    char* readMemoryMapped(int n) override {
        if (m_pos + n > m_size) throw Iex::InputExc("Read past end of EXR buffer");
        char* ptr = const_cast<char*>(reinterpret_cast<const char*>(m_data + m_pos));
        m_pos += n;
        return ptr;
    }

    uint64_t tellg() override {
        return m_pos;
    }
    void seekg(uint64_t pos) override {
        m_pos = static_cast<size_t>(pos);
    }

   private:
    const unsigned char* m_data;
    size_t m_size;
    size_t m_pos;
};

float* load_image_float(const unsigned char* buf, size_t size, const std::string& path, int* w,
                        int* h) {
    if (has_exr_extension(path)) {
        try {
            MemoryIStream stream(path.c_str(), buf, size);
            Imf::InputFile file(stream);
            const auto& header = file.header();
            auto dw = header.dataWindow();
            int width = dw.max.x - dw.min.x + 1;
            int height = dw.max.y - dw.min.y + 1;
            if (width <= 0 || height <= 0) return nullptr;
            *w = width;
            *h = height;

            size_t pixel_count = static_cast<size_t>(width) * height;
            auto* out = static_cast<float*>(std::malloc(pixel_count * 4 * sizeof(float)));
            POSTCONDITION(out);

            // Pre-fill alpha to 1.0
            for (size_t i = 0; i < pixel_count; ++i) out[i * 4 + 3] = 1.0f;

            const auto& channels = header.channels();
            Imf::FrameBuffer fb;
            size_t x_stride = 4 * sizeof(float);
            size_t y_stride = static_cast<size_t>(width) * x_stride;
            char* base = reinterpret_cast<char*>(out) - dw.min.x * static_cast<int64_t>(x_stride) -
                         dw.min.y * static_cast<int64_t>(y_stride);

            struct ChanMap {
                const char* name;
                int offset;
                double fill;
            };
            ChanMap maps[] = {{"R", 0, 0.0}, {"G", 1, 0.0}, {"B", 2, 0.0}, {"A", 3, 1.0}};
            for (auto& m : maps) {
                fb.insert(m.name, Imf::Slice(Imf::FLOAT, base + m.offset * sizeof(float), x_stride,
                                             y_stride, 1, 1, m.fill));
            }

            // Luminance-only fallback
            if (!channels.findChannel("R") && channels.findChannel("Y")) {
                fb.insert("Y", Imf::Slice(Imf::FLOAT, base, x_stride, y_stride));
                file.setFrameBuffer(fb);
                file.readPixels(dw.min.y, dw.max.y);
                for (size_t i = 0; i < pixel_count; ++i) {
                    out[i * 4 + 1] = out[i * 4];
                    out[i * 4 + 2] = out[i * 4];
                }
                return out;
            }

            file.setFrameBuffer(fb);
            file.readPixels(dw.min.y, dw.max.y);
            return out;
        } catch (const std::exception& e) {
            spdlog::warn("EXR decode failed '{}': {}", path, e.what());
            return nullptr;
        }
    }
    // HDR files: use stbi_loadf for true floating-point decode (linear).
    if (has_hdr_extension(path)) {
        int channels = 0;
        float* data = stbi_loadf_from_memory(reinterpret_cast<const stbi_uc*>(buf),
                                             static_cast<int>(size), w, h, &channels, 4);
        if (!data) return nullptr;
        size_t count = static_cast<size_t>(*w) * static_cast<size_t>(*h) * 4;
        auto* out = static_cast<float*>(std::malloc(count * sizeof(float)));
        POSTCONDITION(out);
        std::memcpy(out, data, count * sizeof(float));
        stbi_image_free(data);
        return out;
    }
    // LDR formats (PNG, JPG, etc.): use stbi_load (uint8) and normalize to
    // [0,1] without gamma conversion. stbi_loadf would apply sRGB->linear
    // (pow 2.2), causing double-linearization when the shader also applies it.
    int channels = 0;
    auto* bytes = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(buf),
                                        static_cast<int>(size), w, h, &channels, 4);
    if (!bytes) return nullptr;
    size_t count = static_cast<size_t>(*w) * static_cast<size_t>(*h) * 4;
    auto* out = static_cast<float*>(std::malloc(count * sizeof(float)));
    POSTCONDITION(out);
    constexpr float inv = 1.0f / 255.0f;
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<float>(bytes[i]) * inv;
    }
    stbi_image_free(bytes);
    return out;
}

void resize_rgba_float(const float* src, uint32_t src_w, uint32_t src_h, float* dst, uint32_t dst_w,
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
                dst[(y * dst_w + x) * 4 + c] = p00 * (1 - fx) * (1 - fy) + p10 * fx * (1 - fy) +
                                               p01 * (1 - fx) * fy + p11 * fx * fy;
            }
        }
    }
}

}  // namespace

uint32_t SyncScope::load_texture(const std::string& resolved_path) {
    auto it = m_world.m_texture_cache.find(resolved_path);
    if (it != m_world.m_texture_cache.end()) return it->second;

    auto asset = pxr::ArGetResolver().OpenAsset(pxr::ArResolvedPath(resolved_path));
    if (!asset) {
        spdlog::warn("Failed to open texture asset: {}", resolved_path);
        m_world.m_texture_cache[resolved_path] = UINT32_MAX;
        return UINT32_MAX;
    }

    auto buffer = asset->GetBuffer();
    auto size = asset->GetSize();
    CHECK_MSG(buffer != nullptr, "ArAsset::GetBuffer() returned null for opened asset");

    int w = 0, h = 0;
    float* fdata = load_image_float(reinterpret_cast<const unsigned char*>(buffer.get()),
                                    static_cast<size_t>(size), resolved_path, &w, &h);
    if (!fdata) {
        spdlog::warn("Failed to decode texture: {}", resolved_path);
        m_world.m_texture_cache[resolved_path] = UINT32_MAX;
        return UINT32_MAX;
    }

    auto index = static_cast<uint32_t>(m_world.m_texture_images.size());
    auto tex_size = m_world.m_texture_size;

    float* resized = nullptr;
    float* src = fdata;
    if (static_cast<uint32_t>(w) != tex_size || static_cast<uint32_t>(h) != tex_size) {
        resized = static_cast<float*>(
            std::malloc(static_cast<size_t>(tex_size) * tex_size * 4 * sizeof(float)));
        POSTCONDITION(resized);
        resize_rgba_float(fdata, static_cast<uint32_t>(w), static_cast<uint32_t>(h), resized,
                          tex_size, tex_size);
        src = resized;
    }

    size_t pixel_count = static_cast<size_t>(tex_size) * tex_size * 4;
    RenderWorld::ImageData img;
    img.pixels.resize(pixel_count);
    for (size_t i = 0; i < pixel_count; ++i) {
        img.pixels[i] = float_to_half(src[i]);
    }
    img.width = tex_size;
    img.height = tex_size;

    std::free(resized);
    std::free(fdata);

    m_world.m_texture_images.push_back(std::move(img));
    m_world.m_texture_cache[resolved_path] = index;
    m_world.mark_world_dirty(WorldDirty::SceneTextures);
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
    if (any_world_dirty(m_materials_consumer, WorldDirty::Materials)) {
        data.materials = m_materials;
        data.materials_dirty = true;
        take_world_dirty(m_materials_consumer, WorldDirty::Materials);
    }

    // --- Lights ---
    auto lights_raw = m_lights.span_raw();
    auto lights_cap = static_cast<uint32_t>(m_lights.capacity());

    // Lifecycle bit signals structural change (insert/erase). When set,
    // we do a full rebuild; otherwise, emit per-slot partial updates.
    bool lights_structurally_dirty =
        m_lights.any_dirty_for(m_lights_buffer_consumer, LightField::Lifecycle);
    bool lights_field_dirty = m_lights.any_dirty_for(
        m_lights_buffer_consumer, k_light_buffer_fields & ~LightField::Lifecycle);

    if (lights_structurally_dirty) {
        // Full rebuild
        for (const auto& entry : lights_raw) {
            if (!entry.active) continue;
            data.gpu_lights.push_back(to_light(entry.value));
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
        // Drain ALL bits (including field bits we'd otherwise emit as partials)
        // since the full upload reflects every current value.
        m_lights.drain_dirty_for(m_lights_buffer_consumer, k_light_buffer_fields,
                                 [](uint32_t, const LightData&) {});
    } else if (lights_field_dirty) {
        // Partial updates: GPU buffer is dense (active slots only).
        // Compute gpu_idx by counting active slots up to i.
        uint32_t gpu_idx = 0;
        std::vector<uint32_t> slot_to_gpu(lights_cap, UINT32_MAX);
        for (uint32_t i = 0; i < lights_cap; ++i) {
            if (!lights_raw[i].active) continue;
            slot_to_gpu[i] = gpu_idx++;
        }
        m_lights.drain_dirty_for(
            m_lights_buffer_consumer, k_light_buffer_fields & ~LightField::Lifecycle,
            [&](uint32_t i, const LightData& ld) {
                if (slot_to_gpu[i] == UINT32_MAX) return;
                data.partial_light_updates.push_back({slot_to_gpu[i], to_light(ld)});
            });
    }

    // --- Stamp light_index on proxy materials ---
    if (data.lights_dirty) {
        if (!data.materials_dirty) {
            data.materials = m_materials;
            data.materials_dirty = true;
        }
        for (auto& mat : data.materials) mat.light_index = UINT32_MAX;
        uint32_t gpu_idx = 0;
        for (const auto& entry : lights_raw) {
            if (!entry.active) {
                continue;
            }
            auto mat_idx = entry.value.material_index;
            if (mat_idx != k_no_material &&
                mat_idx < static_cast<uint32_t>(data.materials.size())) {
                data.materials[mat_idx].light_index = gpu_idx;
            }
            ++gpu_idx;
        }
    }

    // --- Two-level BVH (BLAS per mesh, TLAS over instances) ---
    {
        auto objects_raw = m_objects.span_raw();
        auto objects_cap = static_cast<uint32_t>(m_objects.capacity());
        auto meshes_raw = m_meshes.span_raw();
        auto meshes_cap = static_cast<uint32_t>(m_meshes.capacity());

        // Step 1: Collect dirty meshes and pre-populate BLAS cache entries (serial)
        PTS_ZONE_NAMED("BLAS build");
        std::vector<uint32_t> dirty_meshes;
        auto check_mesh_dirty = [&](uint32_t mesh_idx) {
            if (mesh_idx >= meshes_cap) return;
            const auto& mesh_entry = meshes_raw[mesh_idx];
            if (!mesh_entry.active || mesh_entry.value.cpu_vertices.empty() ||
                mesh_entry.value.cpu_indices.empty())
                return;
            auto& blas = m_blas_cache[mesh_idx];
            if (blas.version == mesh_entry.version) return;
            if (std::find(dirty_meshes.begin(), dirty_meshes.end(), mesh_idx) ==
                dirty_meshes.end()) {
                dirty_meshes.push_back(mesh_idx);
            }
        };
        for (uint32_t i = 0; i < objects_cap; ++i) {
            if (!objects_raw[i].active || !objects_raw[i].value.visible) continue;
            check_mesh_dirty(objects_raw[i].value.mesh_index);
        }
        for (uint32_t i = 0; i < lights_cap; ++i) {
            if (!lights_raw[i].active || !lights_raw[i].value.visible ||
                lights_raw[i].value.mesh_index == UINT32_MAX)
                continue;
            check_mesh_dirty(lights_raw[i].value.mesh_index);
        }

        // Build BLAS in parallel (each mesh is independent)
        tbb::parallel_for(tbb::blocked_range<size_t>(0, dirty_meshes.size()),
                          [&](const tbb::blocked_range<size_t>& range) {
                              for (size_t i = range.begin(); i < range.end(); ++i) {
                                  uint32_t mesh_idx = dirty_meshes[i];
                                  auto& blas = m_blas_cache[mesh_idx];
                                  const auto& mesh_entry = meshes_raw[mesh_idx];
                                  blas.tris = blas.bvh.build_from_mesh(
                                      mesh_entry.value.cpu_vertices, mesh_entry.value.cpu_indices);
                                  blas.version = mesh_entry.version;
                              }
                          });
        bool any_blas_dirty = !dirty_meshes.empty();

        // Drain mesh BLAS dirty bits -- the cache version snapshot above is
        // the authoritative gate, but consume the bits so future builds
        // don't re-evaluate already-built BLAS.
        m_meshes.drain_dirty_for(m_meshes_blas_consumer, k_mesh_blas_fields,
                                 [](uint32_t, const MeshData&) {});

        // Step 2: Build instance array + TLAS
        bool tlas_objects_dirty =
            m_objects.any_dirty_for(m_objects_tlas_consumer, k_object_tlas_fields);
        bool tlas_lights_dirty =
            m_lights.any_dirty_for(m_lights_tlas_consumer, k_light_tlas_fields);
        bool need_rebuild = any_blas_dirty || tlas_objects_dirty || tlas_lights_dirty;

        if (need_rebuild) {
            // Collect instances and their world-space AABBs
            struct InstanceInfo {
                uint32_t mesh_index;
                uint32_t material_index;
                glm::mat4 transform;
            };
            std::vector<InstanceInfo> instances;
            std::vector<AABB> world_aabbs;

            for (uint32_t i = 0; i < objects_cap; ++i) {
                const auto& obj_entry = objects_raw[i];
                if (!obj_entry.active) continue;
                if (!obj_entry.value.visible) continue;
                uint32_t mesh_idx = obj_entry.value.mesh_index;
                if (mesh_idx >= meshes_cap) continue;
                const auto& mesh_entry = meshes_raw[mesh_idx];
                if (!mesh_entry.active || mesh_entry.value.cpu_vertices.empty() ||
                    mesh_entry.value.cpu_indices.empty())
                    continue;

                INVARIANT(m_blas_cache.count(mesh_idx) > 0);

                AABB local_aabb = AABB::from_min_max(mesh_entry.value.local_aabb_min,
                                                     mesh_entry.value.local_aabb_max);
                world_aabbs.push_back(transform_aabb(local_aabb, obj_entry.value.transform));
                instances.push_back(
                    {mesh_idx, obj_entry.value.material_index, obj_entry.value.transform});
            }

            // Include light proxy meshes in the BVH so the path tracer
            // can hit emitter geometry (area lights, sphere lights, etc.)
            for (uint32_t i = 0; i < lights_cap; ++i) {
                const auto& light_entry = lights_raw[i];
                if (!light_entry.active) continue;
                if (!light_entry.value.visible) continue;
                if (light_entry.value.mesh_index == UINT32_MAX) continue;
                uint32_t mesh_idx = light_entry.value.mesh_index;
                if (mesh_idx >= meshes_cap) continue;
                const auto& mesh_entry = meshes_raw[mesh_idx];
                if (!mesh_entry.active || mesh_entry.value.cpu_vertices.empty() ||
                    mesh_entry.value.cpu_indices.empty())
                    continue;

                INVARIANT(m_blas_cache.count(mesh_idx) > 0);

                AABB local_aabb = AABB::from_min_max(mesh_entry.value.local_aabb_min,
                                                     mesh_entry.value.local_aabb_max);
                world_aabbs.push_back(transform_aabb(local_aabb, light_entry.value.transform));
                instances.push_back(
                    {mesh_idx, light_entry.value.material_index, light_entry.value.transform});
            }

            auto inst_count = static_cast<uint32_t>(instances.size());

            // Build TLAS from world-space AABBs into PreparedSceneData
            // (not m_tlas -- that's read by the render thread)
            {
                PTS_ZONE_NAMED("TLAS build");
                data.tlas.build(world_aabbs, inst_count);
            }
            uint32_t tlas_nc = data.tlas.node_count();

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
            if (!data.tlas.tri_indices().empty() && inst_count > 0) {
                INVARIANT(data.tlas.tri_indices().size() == inst_count);
                std::vector<GPUInstance> reordered(inst_count);
                for (uint32_t i = 0; i < inst_count; ++i) {
                    reordered[i] = gpu_instances[data.tlas.tri_indices()[i]];
                }
                gpu_instances = std::move(reordered);
            }

            // Concatenate TLAS + BLAS nodes
            std::vector<BlasEntry> blas_entries;
            blas_entries.reserve(unique_meshes.size());
            for (uint32_t mi : unique_meshes) {
                blas_entries.push_back({&m_blas_cache[mi].bvh, mesh_offsets[mi].tri_offset});
            }
            data.all_nodes = data.tlas.concatenate_nodes(blas_entries);

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

            // Drain TLAS-related dirty bits.
            m_objects.drain_dirty_for(m_objects_tlas_consumer, k_object_tlas_fields,
                                      [](uint32_t, const ObjectData&) {});
            m_lights.drain_dirty_for(m_lights_tlas_consumer, k_light_tlas_fields,
                                     [](uint32_t, const LightData&) {});
        }
    }

    // --- Texture array ---
    if (any_world_dirty(m_textures_consumer, WorldDirty::SceneTextures)) {
        data.texture_size = m_texture_size;
        for (const auto& img : m_texture_images) {
            data.texture_layers.push_back({img.pixels.data(), img.width, img.height});
        }
        data.textures_dirty = true;
        take_world_dirty(m_textures_consumer, WorldDirty::SceneTextures);
    }

    return data;
}

void RenderWorld::upload_prepared_data(const webgpu::Device& device, WGPUQueue queue,
                                       PreparedSceneData data) {
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
        ++m_gpu_material_buffer_version;
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
        ++m_gpu_light_buffer_version;
    } else if (!data.partial_light_updates.empty()) {
        for (const auto& update : data.partial_light_updates) {
            wgpuQueueWriteBuffer(queue, m_gpu_light_buffer.handle(),
                                 update.gpu_index * sizeof(Light), &update.data, sizeof(Light));
        }
        ++m_gpu_light_buffer_version;
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

        m_tlas = std::move(data.tlas);
        m_tlas_node_count = data.tlas_node_count;
        m_instance_count = data.instance_count;

        ++m_gpu_bvh_buffer_version;
        ++m_gpu_triangle_buffer_version;
        ++m_gpu_instance_buffer_version;
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
        tex_desc.format = WGPUTextureFormat_RGBA16Float;
        tex_desc.mipLevelCount = 1;
        m_texture_array = wgpuDeviceCreateTexture(device.handle(), &tex_desc);
        POSTCONDITION(m_texture_array);

        if (data.texture_layers.empty()) {
            // 1x1 white placeholder (half-float 1.0 = 0x3C00)
            uint16_t white[] = {0x3C00, 0x3C00, 0x3C00, 0x3C00};
            WGPUTexelCopyTextureInfo dst = {};
            dst.texture = m_texture_array;
            dst.mipLevel = 0;
            dst.origin = {0, 0, 0};
            dst.aspect = WGPUTextureAspect_All;
            WGPUTexelCopyBufferLayout layout = {};
            layout.offset = 0;
            layout.bytesPerRow = 4 * sizeof(uint16_t);
            layout.rowsPerImage = 1;
            WGPUExtent3D extent = {1, 1, 1};
            wgpuQueueWriteTexture(queue, &dst, white, sizeof(white), &layout, &extent);
        } else {
            uint32_t bytes_per_row = tex_w * 4 * sizeof(uint16_t);
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
                wgpuQueueWriteTexture(
                    queue, &dst, layer.pixels,
                    static_cast<std::size_t>(tex_w) * tex_h * 4 * sizeof(uint16_t), &layout,
                    &extent);
            }
        }

        WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
        view_desc.format = WGPUTextureFormat_RGBA16Float;
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

        ++m_gpu_textures_version;
    }
}

void RenderWorld::prepare_gpu_buffers(const webgpu::Device& device, WGPUQueue queue) {
    auto prepared = prepare_scene_data();
    upload_prepared_data(device, queue, std::move(prepared));
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
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_meshes.capacity()); ++i) {
        if (!m_meshes.active_at(i)) continue;
        const auto& mesh = m_meshes.at(i);
        if (mesh.cpu_vertices.empty()) continue;

        PRECONDITION(!mesh.cpu_indices.empty());

        m_meshes.mutate_at(i, MeshField::GpuBuffers | MeshField::Geometry, [&](MeshData& w) {
            w.vertex_buffer = device.create_buffer(
                mesh.cpu_vertices.size() * sizeof(Vertex),
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst));
            wgpuQueueWriteBuffer(device.queue(), w.vertex_buffer.handle(), 0,
                                 mesh.cpu_vertices.data(),
                                 mesh.cpu_vertices.size() * sizeof(Vertex));

            w.index_buffer = device.create_buffer(
                mesh.cpu_indices.size() * sizeof(uint32_t),
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst));
            wgpuQueueWriteBuffer(device.queue(), w.index_buffer.handle(), 0,
                                 mesh.cpu_indices.data(),
                                 mesh.cpu_indices.size() * sizeof(uint32_t));

            w.index_count = static_cast<uint32_t>(mesh.cpu_indices.size());

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
            w.local_aabb_min = aabb_min;
            w.local_aabb_max = aabb_max;
            w.position_buffer = device.create_buffer(
                vert_count * sizeof(glm::vec3),
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst));
            wgpuQueueWriteBuffer(device.queue(), w.position_buffer.handle(), 0, positions.data(),
                                 vert_count * sizeof(glm::vec3));
        });
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
    m_gpu_light_buffer = {};
    m_gpu_material_buffer = {};
    m_gpu_light_count = 0;

    // Two-level BVH state
    m_blas_cache.clear();
    m_tlas = {};
    m_gpu_bvh_nodes = {};
    m_gpu_triangles = {};
    m_gpu_instances = {};
    m_tlas_node_count = 0;
    m_instance_count = 0;

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

    // After clear, every consumer should see a fresh population on the next
    // sync. Re-prime world consumers' pending bits to their full subscription
    // (SlotMap consumers re-prime naturally via on_lifecycle_change on the
    // first insert).
    for (auto& c : m_world_consumers) {
        c.pending = c.subscription;
    }

    // IBL state
    m_ibl = {};
    m_ibl_env_path.clear();
    m_ibl_uniform_color = glm::vec3(-1.0f);
    m_ibl_up_axis = UpAxis::Y;
}

// --- update_transforms ---

void RenderWorld::update_transforms(const pxr::UsdStageRefPtr& stage,
                                    const std::vector<pxr::SdfPath>& dirty_paths) {
    for (const auto& dirty_path : dirty_paths) {
        // Update objects
        m_objects.for_each([&](const pxr::SdfPath& path, const ObjectData&) {
            if (!path.HasPrefix(dirty_path)) return;
            auto prim = stage->GetPrimAtPath(path);
            if (!prim.IsValid()) return;
            auto xf = compute_world_transform(prim);
            auto h = m_objects.find(path);
            m_objects.mutate(h, ObjectField::Transform,
                             [&](ObjectData& obj) { obj.transform = xf; });
        });

        // Update lights. Transform changes propagate to the light buffer
        // consumer (LightField::Transform) AND to the TLAS consumer (which
        // also subscribes to LightField::Transform). The proxy-mesh hack
        // from commit 83351f2 -- bumping m_instances_version explicitly
        // when has_proxy_mesh -- is no longer needed: the TLAS consumer's
        // subscription naturally captures Transform mutations.
        m_lights.for_each([&](const pxr::SdfPath& path, const LightData&) {
            if (!path.HasPrefix(dirty_path)) return;
            auto prim = stage->GetPrimAtPath(path);
            if (!prim.IsValid()) return;
            auto xf = compute_world_transform(prim);
            auto h = m_lights.find(path);
            // For Distant lights we also recompute direction from the new xform.
            auto changed = LightField::Transform;
            if (h->type == LightData::Type::Distant) {
                changed |= LightField::Direction;
            }
            m_lights.mutate(h, changed, [&](LightData& light) {
                light.transform = xf;
                if (light.type == LightData::Type::Distant) {
                    glm::vec4 local_dir(0.0f, 0.0f, -1.0f, 0.0f);
                    light.direction = glm::normalize(glm::vec3(xf * local_dir));
                }
            });
        });

        // Update cameras
        m_cameras.for_each([&](const pxr::SdfPath& path, const CameraData&) {
            if (!path.HasPrefix(dirty_path)) return;
            auto prim = stage->GetPrimAtPath(path);
            if (!prim.IsValid()) return;
            auto xf = compute_world_transform(prim);
            auto h = m_cameras.find(path);
            m_cameras.mutate(h, [&](CameraData& cam) { cam.view_matrix = glm::inverse(xf); });
        });
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

void RenderWorld::update_ibl(const webgpu::Device& device, WGPUQueue queue, WGPUSampler ibl_sampler,
                             UpAxis up_axis) {
    PTS_ZONE_SCOPED;

    // Lazy-init pipelines on first call
    if (!m_ibl_pipelines) {
        m_ibl_pipelines = std::make_unique<IblPipelines>();
        m_ibl_pipelines->init(device, queue, ibl_sampler);
    }

    // Re-evaluate when dome-light state changes (Type/Color/Intensity/EnvTexture)
    // or when up_axis changes.
    bool lights_dirty = m_lights.any_dirty_for(m_lights_ibl_consumer, k_light_ibl_fields);
    bool axis_changed = (up_axis != m_ibl_up_axis);
    if (!lights_dirty && !axis_changed) return;

    // Find first dome light
    const LightData* dome = nullptr;
    auto lights_raw = m_lights.span_raw();
    for (const auto& entry : lights_raw) {
        if (!entry.active) continue;
        if (entry.value.type == LightData::Type::Dome) {
            dome = &entry.value;
            break;
        }
    }

    auto drain_dome_dirty = [&]() {
        m_lights.drain_dirty_for(m_lights_ibl_consumer, k_light_ibl_fields,
                                 [](uint32_t, const LightData&) {});
    };

    if (!dome) {
        // No dome light -- black ambient
        if (m_ibl_env_path.empty() && m_ibl_uniform_color == glm::vec3(0.0f)) {
            drain_dome_dirty();
            return;
        }
        m_ibl.set_uniform_environment(device, queue, 0.0f, 0.0f, 0.0f);
        m_ibl_env_path.clear();
        m_ibl_uniform_color = glm::vec3(0.0f);
        drain_dome_dirty();
        return;
    }

    if (!dome->env_texture_path.empty()) {
        // HDR environment map
        if (dome->env_texture_path == m_ibl_env_path && up_axis == m_ibl_up_axis) {
            drain_dome_dirty();
            return;
        }

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

        int w = 0, h = 0;
        float* data = load_image_float(reinterpret_cast<const unsigned char*>(buffer.get()),
                                       static_cast<size_t>(size), dome->env_texture_path, &w, &h);
        if (!data) {
            spdlog::warn("Failed to decode HDR environment: {}", dome->env_texture_path);
            return;
        }

        m_ibl.set_environment(*m_ibl_pipelines, device, queue, data, static_cast<uint32_t>(w),
                              static_cast<uint32_t>(h), up_axis);
        std::free(data);

        m_ibl_env_path = dome->env_texture_path;
        m_ibl_up_axis = up_axis;
        m_ibl_uniform_color = glm::vec3(-1.0f);  // invalidate uniform sentinel
    } else {
        // Uniform color environment: dome color * intensity
        glm::vec3 c = dome->color * dome->intensity;
        if (m_ibl_env_path.empty() && m_ibl_uniform_color == c) {
            drain_dome_dirty();
            return;
        }

        m_ibl.set_uniform_environment(device, queue, c.r, c.g, c.b);
        m_ibl_env_path.clear();
        m_ibl_uniform_color = c;
    }

    drain_dome_dirty();
}

}  // namespace pts::rendering
