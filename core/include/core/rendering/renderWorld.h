#pragma once

#include <core/container/slotArray.h>
#include <core/container/slotMap.h>
#include <core/diagnostics.h>
#include <core/rendering/bvh.h>
#include <core/rendering/iblResources.h>
#include <core/rendering/packedTriangle.h>
#include <core/rendering/versionedBuffer.h>
#include <core/rendering/vertex.h>
#include <core/rendering/webgpu/buffer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
#include <webgpu/webgpu.h>

#include <boost/container/flat_map.hpp>
#include <boost/core/span.hpp>
#include <climits>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace pts::webgpu {
class Device;
}

namespace pts::rendering {

static constexpr uint32_t k_no_material = UINT32_MAX;
static constexpr uint32_t k_default_material = 0;

/// Per-field dirty bits for Material. Used by the materials SlotArray to
/// notify consumers (GPU buffer rebuild) of targeted changes. The `All`
/// value covers future fields added below.
enum class MaterialField : uint32_t {
    None = 0,
    Albedo = 1u << 0,        ///< diffuse_color
    Metallic = 1u << 1,      ///< metallic
    Roughness = 1u << 2,     ///< roughness
    Emissive = 1u << 3,      ///< emissive_color
    Transmission = 1u << 4,  ///< opacity / opacity_threshold
    Ior = 1u << 5,           ///< ior
    Textures = 1u << 6,      ///< any texture slot or channel mapping
    LightIndex = 1u << 7,    ///< light_index stamping from proxy emitters
    All = ~0u,
};
constexpr MaterialField operator|(MaterialField a, MaterialField b) noexcept {
    return static_cast<MaterialField>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr MaterialField operator&(MaterialField a, MaterialField b) noexcept {
    return static_cast<MaterialField>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr MaterialField operator~(MaterialField a) noexcept {
    return static_cast<MaterialField>(~static_cast<uint32_t>(a));
}
constexpr MaterialField& operator|=(MaterialField& a, MaterialField b) noexcept {
    a = a | b;
    return a;
}
constexpr MaterialField& operator&=(MaterialField& a, MaterialField b) noexcept {
    a = a & b;
    return a;
}

/// Per-field dirty bits for scene textures. Textures today are load-once,
/// never mutated; the single Pixels bit is enough for current needs.
enum class TextureField : uint32_t {
    None = 0,
    Pixels = 1u << 0,  ///< pixel data (RGBA16Float tile)
    All = ~0u,
};
constexpr TextureField operator|(TextureField a, TextureField b) noexcept {
    return static_cast<TextureField>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr TextureField operator&(TextureField a, TextureField b) noexcept {
    return static_cast<TextureField>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr TextureField operator~(TextureField a) noexcept {
    return static_cast<TextureField>(~static_cast<uint32_t>(a));
}
constexpr TextureField& operator|=(TextureField& a, TextureField b) noexcept {
    a = a | b;
    return a;
}
constexpr TextureField& operator&=(TextureField& a, TextureField b) noexcept {
    a = a & b;
    return a;
}

/// 80-byte GPU struct
struct Material {
    glm::vec3 diffuse_color{1.0f, 1.0f, 1.0f};
    float metallic{0.0f};
    glm::vec3 emissive_color{0.0f, 0.0f, 0.0f};
    float roughness{0.5f};
    float opacity{1.0f};
    uint32_t diffuse_tex{UINT32_MAX};
    uint32_t normal_tex{UINT32_MAX};
    uint32_t metallic_tex{UINT32_MAX};
    uint32_t roughness_tex{UINT32_MAX};
    uint32_t emissive_tex{UINT32_MAX};
    uint32_t opacity_tex{UINT32_MAX};
    float ior{1.5f};
    float opacity_threshold{0.0f};
    uint32_t tex_channels{0};
    uint32_t light_index{UINT32_MAX};  // GPU light index for proxy meshes (UINT32_MAX = none)
    uint32_t _pad{};
};
static_assert(sizeof(Material) == 80, "Material must be 80 bytes for GPU alignment");

/// 64-byte GPU struct
struct Light {
    glm::vec3 direction_or_pos;
    uint32_t type;
    glm::vec3 color;
    float intensity;
    glm::vec3 right;
    float radius;
    glm::vec3 up;
    float angle;
};
static_assert(sizeof(Light) == 64, "Light must be 64 bytes for GPU alignment");

// --- Data structs ---

struct MeshData {
    webgpu::Buffer vertex_buffer;  // interleaved (pos+normal+color+mat_idx)
    webgpu::Buffer index_buffer;
    webgpu::Buffer position_buffer;  // position-only (for picking, depth prepass)
    uint32_t index_count = 0;
    std::vector<uint32_t> cpu_indices;
    std::vector<Vertex> cpu_vertices;
    glm::vec3 local_aabb_min{0};
    glm::vec3 local_aabb_max{0};
};

/// Per-field dirty bits for MeshData. Insert stamps the new slot with the
/// full subscription mask automatically (see SlotArray docs); erase queues
/// an erase event observable via `drain()`'s on_erase callback.
enum class MeshField : uint32_t {
    None = 0,
    Geometry = 1u << 1,    ///< cpu_vertices, cpu_indices, AABB
    GpuBuffers = 1u << 2,  ///< vertex_buffer, index_buffer, position_buffer
    All = ~0u,
};
constexpr MeshField operator|(MeshField a, MeshField b) noexcept {
    return static_cast<MeshField>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr MeshField operator&(MeshField a, MeshField b) noexcept {
    return static_cast<MeshField>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr MeshField operator~(MeshField a) noexcept {
    return static_cast<MeshField>(~static_cast<uint32_t>(a));
}
constexpr MeshField& operator|=(MeshField& a, MeshField b) noexcept {
    a = a | b;
    return a;
}
constexpr MeshField& operator&=(MeshField& a, MeshField b) noexcept {
    a = a & b;
    return a;
}

struct ObjectData {
    uint32_t mesh_index = 0;
    uint32_t material_index{k_no_material};
    glm::mat4 transform{1.0f};
    bool visible{true};
};

enum class ObjectField : uint32_t {
    None = 0,
    Transform = 1u << 1,
    Visibility = 1u << 2,
    MeshIndex = 1u << 3,
    MaterialIndex = 1u << 4,
    All = ~0u,
};
constexpr ObjectField operator|(ObjectField a, ObjectField b) noexcept {
    return static_cast<ObjectField>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr ObjectField operator&(ObjectField a, ObjectField b) noexcept {
    return static_cast<ObjectField>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr ObjectField operator~(ObjectField a) noexcept {
    return static_cast<ObjectField>(~static_cast<uint32_t>(a));
}
constexpr ObjectField& operator|=(ObjectField& a, ObjectField b) noexcept {
    a = a | b;
    return a;
}
constexpr ObjectField& operator&=(ObjectField& a, ObjectField b) noexcept {
    a = a & b;
    return a;
}

struct LightData {
    enum class Type { Distant, Sphere, Rect, Disk, Dome };
    Type type = Type::Distant;
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    float intensity{1.0f};
    glm::mat4 transform{1.0f};
    glm::vec3 direction{0.0f, -1.0f, 0.0f};
    float angle{0.53f};
    float radius{0.0f};
    float width{1.0f};
    float height{1.0f};
    bool casts_shadow{true};          // from UsdLuxShadowAPI inputs:shadow:enable
    std::string env_texture_path;     // resolved path to HDR environment map (dome lights only)
    uint32_t mesh_index{UINT32_MAX};  // proxy mesh slot (UINT32_MAX = none)
    uint32_t material_index{k_no_material};  // emissive material index
    bool visible{true};                      // USD visibility for proxy mesh
};

enum class LightField : uint32_t {
    None = 0,
    Transform = 1u << 1,
    Color = 1u << 2,
    Intensity = 1u << 3,
    Type = 1u << 4,
    Visibility = 1u << 5,
    CastsShadow = 1u << 6,
    MeshIndex = 1u << 7,
    MaterialIndex = 1u << 8,
    EnvTexture = 1u << 9,
    Direction = 1u << 10,
    Geometry = 1u << 11,  ///< angle, radius, width, height
    All = ~0u,
};
constexpr LightField operator|(LightField a, LightField b) noexcept {
    return static_cast<LightField>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr LightField operator&(LightField a, LightField b) noexcept {
    return static_cast<LightField>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr LightField operator~(LightField a) noexcept {
    return static_cast<LightField>(~static_cast<uint32_t>(a));
}
constexpr LightField& operator|=(LightField& a, LightField b) noexcept {
    a = a | b;
    return a;
}
constexpr LightField& operator&=(LightField& a, LightField b) noexcept {
    a = a & b;
    return a;
}

/// Convert a LightData to a GPU-ready Light struct.
Light to_light(const LightData& slot);

struct CameraData {
    glm::mat4 view_matrix{1.0f};
    float fov_y_radians{0.8f};
    float ortho_height{10.0f};
    float near_clip{0.1f};
    float far_clip{10000.0f};
    bool orthographic{false};
};

/// Prim path -> slot lookup entry.
struct PrimSlot {
    enum class Kind : uint8_t { Object, Light, Camera };
    Kind kind;
    uint32_t index;
};

/// 96-byte per-light shadow info (one entry per light in the light buffer).
/// Lights without shadows have has_shadow == 0. The near/far planes and
/// light_size_uv feed PCSS soft-shadow sampling (shadow_sampling.slang).
struct ShadowInfo {
    glm::mat4 light_vp{1.0f};      // 64 bytes
    float texel_size = 0.0f;       //  4 bytes
    float normal_bias = 0.0f;      //  4 bytes
    uint32_t has_shadow = 0;       //  4 bytes -- 0 = no shadow, 1 = active
    uint32_t layer = 0;            //  4 bytes -- texture array layer index
    float light_near = 0.0f;       //  4 bytes -- light-space near plane (for linear-depth recon)
    float light_far = 0.0f;        //  4 bytes -- light-space far plane
    float light_size_uv = 0.0f;    //  4 bytes -- PCSS light size (see shadow_sampling.slang)
    uint32_t projection_type = 0;  //  4 bytes -- 0 = ortho (distant), 1 = perspective (area)
};
static_assert(sizeof(ShadowInfo) == 96, "ShadowInfo must be 96 bytes for GPU alignment");

/// Per-instance data for two-level BVH traversal on the GPU.
struct GPUInstance {
    glm::mat4 transform{1.0f};      // 64 bytes: object-to-world
    glm::mat4 inv_transform{1.0f};  // 64 bytes: world-to-object
    uint32_t blas_offset{0};        // absolute index into bvh_nodes buffer
    uint32_t tri_offset{0};         // offset into global triangle buffer
    uint32_t tri_count{0};          // triangles in this mesh
    uint32_t material_index{UINT32_MAX};
};
static_assert(sizeof(GPUInstance) == 144);

// SlotMap type aliases for world data. Lights/objects/meshes carry per-field
// dirty masks; cameras don't (they aren't multiplexed across cache consumers).
using ObjectSlotMap = container::SlotMap<pxr::SdfPath, ObjectData, ObjectField>;
using MeshSlotMap = container::SlotMap<pxr::SdfPath, MeshData, MeshField>;
using LightSlotMap = container::SlotMap<pxr::SdfPath, LightData, LightField>;
using CameraSlotMap = container::SlotMap<pxr::SdfPath, CameraData>;

struct RenderWorld;
struct PreparedSceneData;

/// Texture pixel payload stored in the scene textures SlotArray. Kept as a
/// public type on RenderWorld's namespace so SlotArray template users can
/// reference it without reaching into RenderWorld's private scope.
struct SceneTexture {
    std::vector<uint16_t> pixels;  // RGBA16Float (half-precision)
    uint32_t width = 0;
    uint32_t height = 0;
};

using MaterialSlotArray = container::SlotArray<Material, MaterialField>;
using TextureSlotArray = container::SlotArray<SceneTexture, TextureField>;

/// RAII scope guard for batched sync operations on a RenderWorld. Per-slot
/// dirty bits on the SlotArray-backed containers (materials, textures,
/// lights, objects, meshes) are driven by the `changed` mask passed to
/// mutate_*() / mutate_at(). No implicit end-of-scope dirty fanout.
class SyncScope {
   public:
    explicit SyncScope(RenderWorld& world);
    ~SyncScope() = default;
    SyncScope(const SyncScope&) = delete;
    SyncScope& operator=(const SyncScope&) = delete;
    SyncScope(SyncScope&&) = delete;
    SyncScope& operator=(SyncScope&&) = delete;

    RenderWorld& world() {
        return m_world;
    }
    const RenderWorld& world() const {
        return m_world;
    }

    /// Allocate a new slot keyed by prim path. Returns the stable index.
    uint32_t alloc_object(const pxr::SdfPath& path);
    uint32_t alloc_mesh(const pxr::SdfPath& path);
    uint32_t alloc_light(const pxr::SdfPath& path);
    uint32_t alloc_camera(const pxr::SdfPath& path);

    /// Erase a slot by prim path.
    void free_object(const pxr::SdfPath& path);
    void free_mesh(const pxr::SdfPath& path);
    void free_light(const pxr::SdfPath& path);
    void free_camera(const pxr::SdfPath& path);

    /// In-place mutation; the `changed` mask declares which fields the
    /// callback touched. Consumers subscribed to those bits will see them
    /// dirty for this slot.
    template <class Fn>
    void mutate_object(uint32_t i, ObjectField changed, Fn&& fn);
    template <class Fn>
    void mutate_mesh(uint32_t i, MeshField changed, Fn&& fn);
    template <class Fn>
    void mutate_light(uint32_t i, LightField changed, Fn&& fn);
    /// Cameras don't drive cached GPU state, so they don't take a mask.
    template <class Fn>
    void mutate_camera(uint32_t i, Fn&& fn);

    template <class Fn>
    void mutate_material(uint32_t i, MaterialField changed, Fn&& fn);

    // Read-only accessors by index.
    const ObjectData& object(uint32_t i) const;
    const MeshData& mesh(uint32_t i) const;
    const LightData& light(uint32_t i) const;
    const CameraData& camera(uint32_t i) const;
    const Material& material(uint32_t i) const;

    MaterialSlotArray& materials();
    std::unordered_map<std::string, uint32_t>& material_cache();

    /// Load a texture from disk, deduplicate by path. Returns layer index
    /// or UINT32_MAX on failure. Marks the scene texture SlotArray's
    /// consumers dirty for the new layer.
    uint32_t load_texture(const std::string& resolved_path);

   private:
    RenderWorld& m_world;
};

struct RenderWorld {
    RenderWorld();

    // Read-only accessors returning const references to SlotMaps.
    const ObjectSlotMap& get_objects() const;
    const MeshSlotMap& get_meshes() const;
    const LightSlotMap& get_lights() const;
    const CameraSlotMap& get_cameras() const;
    const MaterialSlotArray& get_materials_array() const;

    /// User-material snapshot (skips the reserved default material at
    /// slot 0). Returned by value -- use `get_materials_array()` for
    /// zero-copy traversal.
    std::vector<Material> get_materials() const;

    int find_object_by_prim(const pxr::SdfPath& path) const;
    int find_light_by_prim(const pxr::SdfPath& path) const;
    int find_camera_by_prim(const pxr::SdfPath& path) const;

    /// Iterate all prim slots across objects, lights, and cameras.
    /// fn(const pxr::SdfPath& path, PrimSlot slot)
    template <typename F>
    void for_each_prim(F&& fn) const {
        m_objects.for_each([&](const pxr::SdfPath& path, const ObjectData&) {
            fn(path, PrimSlot{PrimSlot::Kind::Object, m_objects.find(path).index()});
        });
        m_lights.for_each([&](const pxr::SdfPath& path, const LightData&) {
            fn(path, PrimSlot{PrimSlot::Kind::Light, m_lights.find(path).index()});
        });
        m_cameras.for_each([&](const pxr::SdfPath& path, const CameraData&) {
            fn(path, PrimSlot{PrimSlot::Kind::Camera, m_cameras.find(path).index()});
        });
    }

    // GPU buffer management

    /// CPU-only: compute all scene data. No GPU calls.
    [[nodiscard]] PreparedSceneData prepare_scene_data();

    /// GPU-only: upload a PreparedSceneData snapshot to GPU buffers.
    void upload_prepared_data(const webgpu::Device& device, WGPUQueue queue,
                              PreparedSceneData data);

    void prepare_gpu_buffers(const webgpu::Device& device, WGPUQueue queue);
    ImportedBuffer light_buffer() const noexcept;
    ImportedBuffer material_buffer() const noexcept;
    uint32_t gpu_light_count() const;
    WGPUTextureView texture_array_view() const;
    WGPUSampler texture_sampler() const;

    /// World-space scene AABB (TLAS root bounds).
    AABB scene_bounds() const;

    /// Concatenated TLAS + BLAS node buffer for GPU traversal.
    ImportedBuffer bvh_node_buffer() const noexcept;

    /// Concatenated local-space triangle buffer (ordered by per-mesh BLAS).
    ImportedBuffer triangle_buffer() const noexcept;

    /// GPUInstance array (reordered by TLAS).
    ImportedBuffer instance_buffer() const noexcept;

    /// Number of TLAS nodes (first N nodes in the concatenated BVH buffer).
    uint32_t tlas_node_count() const;

    /// Number of active instances.
    uint32_t instance_count() const;

    // --- Per-pass data cache (owned by the world, destroyed on world swap) ---
    using ErasedPtr = std::unique_ptr<void, void (*)(void*)>;
    struct PassDataEntry {
        ErasedPtr data{nullptr, nullptr};
        uint32_t version = UINT32_MAX;
    };
    using PassDataMap = boost::container::flat_map<uint64_t, PassDataEntry>;

    /// Get or create the per-pass cache for a given pass instance.
    /// The void* key is typically the pass's `this` pointer.
    PassDataMap& pass_data_for(const void* pass_id) {
        return m_pass_data_cache[pass_id];
    }

    /// Lightweight xform-only update: recomputes world transforms for all
    /// synced prims at or under the given paths. Does not re-upload meshes.
    void update_transforms(const pxr::UsdStageRefPtr& stage,
                           const std::vector<pxr::SdfPath>& dirty_paths);

    /// Update IBL resources from the current dome light state.
    /// Inits BRDF LUT on first call, then loads HDR or sets uniform color.
    /// The sampler is a trilinear-clamp sampler (e.g. from FrameGraph::sampler()).
    void update_ibl(const webgpu::Device& device, WGPUQueue queue, WGPUSampler ibl_sampler,
                    UpAxis up_axis = UpAxis::Y);

    IblResources& ibl_resources();
    const IblResources& ibl_resources() const;
    const IblPipelines& ibl_pipelines() const;

    /// Begin a batched sync operation. mesh_version is bumped when
    /// the returned scope guard is destroyed. sync_object/remove_prim
    /// calls without a live SyncScope will PRECONDITION-fail.
    [[nodiscard]] SyncScope begin_sync();

    /// Upload GPU buffers for all meshes that have CPU vertex data.
    /// Call on the main thread after building the RenderWorld off-thread.
    void upload_all_meshes(const webgpu::Device& device);

    void clear();

   private:
    friend class SyncScope;

    void register_internal_consumers();

    MeshSlotMap m_meshes;
    ObjectSlotMap m_objects;
    MaterialSlotArray m_materials;
    LightSlotMap m_lights;
    CameraSlotMap m_cameras;

    /// Material path -> material index (deduplication cache).
    std::unordered_map<std::string, uint32_t> m_material_cache;

    // --- Internal SlotArray consumer ids (registered in register_internal_consumers) ---
    // Each consumer corresponds to one cached GPU resource maintained by
    // prepare_scene_data / upload_prepared_data.
    LightSlotMap::ConsumerId m_lights_buffer_consumer = 0;
    LightSlotMap::ConsumerId m_lights_tlas_consumer = 0;
    LightSlotMap::ConsumerId m_lights_ibl_consumer = 0;
    ObjectSlotMap::ConsumerId m_objects_tlas_consumer = 0;
    MeshSlotMap::ConsumerId m_meshes_blas_consumer = 0;
    MaterialSlotArray::ConsumerId m_materials_consumer = 0;
    TextureSlotArray::ConsumerId m_textures_consumer = 0;

    // --- GPU buffer state ---
    // VersionedBuffer auto-bumps its internal version on each write, so the
    // (handle, size, version) triple handed to FrameGraph::import_buffer stays
    // in sync with the buffer contents without any manual counter plumbing.
    VersionedBuffer<Light> m_gpu_light_buffer;
    VersionedBuffer<Material> m_gpu_material_buffer;
    uint32_t m_gpu_light_count = 0;

    // Two-level acceleration structure
    struct BlasData {
        BVH bvh;                           // local-space BVH tree
        std::vector<PackedTriangle> tris;  // local-space triangles (BVH-reordered)
        uint64_t version = UINT64_MAX;     // mesh slot version when built
    };
    std::unordered_map<uint32_t, BlasData> m_blas_cache;

    BVH m_tlas;                                       // world-space TLAS
    VersionedBuffer<BVHNode> m_gpu_bvh_nodes;         // concatenated TLAS + BLAS nodes
    VersionedBuffer<PackedTriangle> m_gpu_triangles;  // concatenated local-space triangles
    VersionedBuffer<GPUInstance> m_gpu_instances;     // GPUInstance array
    uint32_t m_tlas_node_count = 0;
    uint32_t m_instance_count = 0;

    // Scene texture state. Stored as a SlotArray so the GPU-upload
    // consumer uses the same dirty-tracking plumbing as materials.
    TextureSlotArray m_texture_images;
    std::unordered_map<std::string, uint32_t> m_texture_cache;
    WGPUTexture m_texture_array = nullptr;
    WGPUTextureView m_texture_array_view = nullptr;
    WGPUSampler m_texture_sampler = nullptr;
    uint32_t m_texture_size = 1024;

    // Per-pass data cache -- keyed by pass identity (this pointer)
    std::unordered_map<const void*, PassDataMap> m_pass_data_cache;

    // IBL state
    std::unique_ptr<IblPipelines> m_ibl_pipelines;
    IblResources m_ibl;
    std::string m_ibl_env_path;            // currently loaded HDR path (empty = uniform)
    glm::vec3 m_ibl_uniform_color{-1.0f};  // sentinel: never matches real color
    UpAxis m_ibl_up_axis = UpAxis::Y;      // up axis when IBL was last converted
};

// SyncScope mutate_* template definitions -- deferred until after
// RenderWorld is complete (see note at the forward declarations above).
template <class Fn>
void SyncScope::mutate_object(uint32_t i, ObjectField changed, Fn&& fn) {
    m_world.m_objects.mutate_at(i, changed, std::forward<Fn>(fn));
}
template <class Fn>
void SyncScope::mutate_mesh(uint32_t i, MeshField changed, Fn&& fn) {
    m_world.m_meshes.mutate_at(i, changed, std::forward<Fn>(fn));
}
template <class Fn>
void SyncScope::mutate_light(uint32_t i, LightField changed, Fn&& fn) {
    m_world.m_lights.mutate_at(i, changed, std::forward<Fn>(fn));
}
template <class Fn>
void SyncScope::mutate_camera(uint32_t i, Fn&& fn) {
    m_world.m_cameras.mutate_at(i, std::forward<Fn>(fn));
}
template <class Fn>
void SyncScope::mutate_material(uint32_t i, MaterialField changed, Fn&& fn) {
    m_world.m_materials.mutate_at(i, changed, std::forward<Fn>(fn));
}

}  // namespace pts::rendering
