#pragma once

#include <core/container/slotMap.h>
#include <core/diagnostics.h>
#include <core/rendering/bvh.h>
#include <core/rendering/iblResources.h>
#include <core/rendering/packedTriangle.h>
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
#include <unordered_map>
#include <vector>

namespace pts::webgpu {
class Device;
}

namespace pts::rendering {

static constexpr uint32_t k_no_material = UINT32_MAX;
static constexpr uint32_t k_default_material = 0;

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

struct ObjectData {
    uint32_t mesh_index = 0;
    uint32_t material_index{k_no_material};
    glm::mat4 transform{1.0f};
    bool visible{true};
};

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

/// 80-byte per-light shadow info (one entry per light in the light buffer).
/// Lights without shadows have has_shadow == 0.
struct ShadowInfo {
    glm::mat4 light_vp{1.0f};  // 64 bytes
    float texel_size = 0.0f;   //  4 bytes
    float normal_bias = 0.0f;  //  4 bytes
    uint32_t has_shadow = 0;   //  4 bytes -- 0 = no shadow, 1 = active
    uint32_t layer = 0;        //  4 bytes -- texture array layer index
};
static_assert(sizeof(ShadowInfo) == 80, "ShadowInfo must be 80 bytes for GPU alignment");

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

// SlotMap type aliases for world data.
using ObjectSlotMap = container::SlotMap<pxr::SdfPath, ObjectData>;
using MeshSlotMap = container::SlotMap<pxr::SdfPath, MeshData>;
using LightSlotMap = container::SlotMap<pxr::SdfPath, LightData>;
using CameraSlotMap = container::SlotMap<pxr::SdfPath, CameraData>;

struct RenderWorld;
struct PreparedSceneData;

/// RAII scope guard for batched sync operations. Bumps mesh_version
/// on destruction. All sync_object/remove_prim calls must happen
/// within a live SyncScope.
class SyncScope {
   public:
    explicit SyncScope(RenderWorld& world);
    ~SyncScope();
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

    /// In-place mutation; bumps version after fn returns.
    template <class Fn>
    void mutate_object(uint32_t i, Fn&& fn) {
        m_world.m_objects.mutate_at(i, std::forward<Fn>(fn));
    }
    template <class Fn>
    void mutate_mesh(uint32_t i, Fn&& fn) {
        m_world.m_meshes.mutate_at(i, std::forward<Fn>(fn));
    }
    template <class Fn>
    void mutate_light(uint32_t i, Fn&& fn) {
        m_world.m_lights.mutate_at(i, std::forward<Fn>(fn));
    }
    template <class Fn>
    void mutate_camera(uint32_t i, Fn&& fn) {
        m_world.m_cameras.mutate_at(i, std::forward<Fn>(fn));
    }

    // Read-only accessors by index.
    const ObjectData& object(uint32_t i) const;
    const MeshData& mesh(uint32_t i) const;
    const LightData& light(uint32_t i) const;
    const CameraData& camera(uint32_t i) const;

    Material& material(uint32_t i);
    std::vector<Material>& materials();
    std::unordered_map<std::string, uint32_t>& material_cache();

    /// Load a texture from disk, deduplicate by path. Returns layer index
    /// or UINT32_MAX on failure. Bumps texture version.
    uint32_t load_texture(const std::string& resolved_path);

   private:
    RenderWorld& m_world;
};

struct RenderWorld {
    RenderWorld() {
        m_materials.push_back(Material{});
    }

    // Read-only accessors returning const references to SlotMaps.
    const ObjectSlotMap& get_objects() const;
    const MeshSlotMap& get_meshes() const;
    const LightSlotMap& get_lights() const;
    const CameraSlotMap& get_cameras() const;
    boost::span<const Material> get_materials() const;

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
    const webgpu::Buffer& light_buffer() const;
    const webgpu::Buffer& material_buffer() const;
    uint32_t gpu_light_count() const;
    WGPUTextureView texture_array_view() const;
    WGPUSampler texture_sampler() const;

    /// World-space scene AABB (TLAS root bounds).
    AABB scene_bounds() const;

    /// Concatenated TLAS + BLAS node buffer for GPU traversal.
    const webgpu::Buffer& bvh_node_buffer() const;

    /// Concatenated local-space triangle buffer (ordered by per-mesh BLAS).
    const webgpu::Buffer& triangle_buffer() const;

    /// GPUInstance array (reordered by TLAS).
    const webgpu::Buffer& instance_buffer() const;

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

    // Category version counters -- bumped by SyncScope when any slot in that
    // category changes.  Used internally by IPass::get_or_create_pass_data
    // and prepare_gpu_buffers.  Prefer the pass_data API over reading these
    // directly in renderer code.
    uint32_t get_mesh_version() const;
    uint32_t get_light_version() const;
    uint32_t get_material_version() const;

    /// Per-kind monotonic version accessors. uint64_t to avoid wraparound.
    /// Dependents (e.g. FG import_buffer with external_version) pass these
    /// into DepTrackedSlotMap deps so bind groups rebuild on world mutations
    /// affecting the bound buffers.
    uint64_t lights_version() const {
        return m_lights_version;
    }
    uint64_t materials_version() const {
        return m_materials_version;
    }
    uint64_t scene_textures_version() const {
        return m_scene_textures_version;
    }
    uint64_t instances_version() const {
        return m_instances_version;
    }
    uint64_t triangles_version() const {
        return m_triangles_version;
    }
    uint64_t bvh_version() const {
        return m_bvh_version;
    }

   private:
    friend class SyncScope;

    MeshSlotMap m_meshes;
    ObjectSlotMap m_objects;
    std::vector<Material> m_materials;
    LightSlotMap m_lights;
    CameraSlotMap m_cameras;

    /// Material path -> material index (deduplication cache).
    std::unordered_map<std::string, uint32_t> m_material_cache;

    uint32_t m_mesh_version = 0;
    // Per-kind monotonic versions. Bumped at mutation points. uint64_t to
    // avoid wraparound across long sessions.
    uint64_t m_lights_version = 0;
    uint64_t m_materials_version = 0;
    uint64_t m_scene_textures_version = 0;
    uint64_t m_instances_version = 0;
    uint64_t m_triangles_version = 0;
    uint64_t m_bvh_version = 0;

    // GPU buffer state
    webgpu::Buffer m_gpu_light_buffer;
    webgpu::Buffer m_gpu_material_buffer;
    uint32_t m_gpu_light_count = 0;
    uint64_t m_cached_lights_version = UINT64_MAX;
    uint64_t m_cached_materials_version = UINT64_MAX;

    // Per-slot version cache for partial light updates
    std::vector<uint64_t> m_cached_light_versions;

    // Two-level acceleration structure
    struct BlasData {
        BVH bvh;                           // local-space BVH tree
        std::vector<PackedTriangle> tris;  // local-space triangles (BVH-reordered)
        uint64_t version = UINT64_MAX;     // mesh slot version when built
    };
    std::unordered_map<uint32_t, BlasData> m_blas_cache;

    BVH m_tlas;                      // world-space TLAS
    webgpu::Buffer m_gpu_bvh_nodes;  // concatenated TLAS + BLAS nodes
    webgpu::Buffer m_gpu_triangles;  // concatenated local-space triangles
    webgpu::Buffer m_gpu_instances;  // GPUInstance array
    uint32_t m_tlas_node_count = 0;
    uint32_t m_instance_count = 0;
    uint64_t m_cached_instances_version = UINT64_MAX;
    uint32_t m_cached_geometry_version = UINT32_MAX;

    // Texture array state
    struct ImageData {
        std::vector<uint16_t> pixels;  // RGBA16Float (half-precision)
        uint32_t width;
        uint32_t height;
    };
    std::vector<ImageData> m_texture_images;
    std::unordered_map<std::string, uint32_t> m_texture_cache;
    WGPUTexture m_texture_array = nullptr;
    WGPUTextureView m_texture_array_view = nullptr;
    WGPUSampler m_texture_sampler = nullptr;
    uint64_t m_cached_scene_textures_version = UINT64_MAX;
    uint32_t m_texture_size = 1024;

    // Per-pass data cache -- keyed by pass identity (this pointer)
    std::unordered_map<const void*, PassDataMap> m_pass_data_cache;

    // IBL state
    std::unique_ptr<IblPipelines> m_ibl_pipelines;
    IblResources m_ibl;
    std::string m_ibl_env_path;                 // currently loaded HDR path (empty = uniform)
    uint64_t m_ibl_light_version = UINT64_MAX;  // light version when IBL was last updated
    glm::vec3 m_ibl_uniform_color{-1.0f};       // sentinel: never matches real color
    UpAxis m_ibl_up_axis = UpAxis::Y;           // up axis when IBL was last converted
};

}  // namespace pts::rendering
