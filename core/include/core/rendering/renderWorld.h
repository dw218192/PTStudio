#pragma once

#include <core/diagnostics.h>
#include <core/rendering/vertex.h>
#include <core/rendering/webgpu/buffer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>

#include <boost/container/flat_map.hpp>
#include <boost/core/span.hpp>
#include <climits>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pts::webgpu {
class Device;
}

namespace pts::rendering {

static constexpr uint32_t k_no_material = UINT32_MAX;

/// 32-byte GPU struct
struct Material {
    glm::vec3 diffuse_color{1.0f, 1.0f, 1.0f};
    float metallic{0.0f};
    float roughness{0.5f};
    float opacity{1.0f};
    uint32_t _padding[2]{};
};
static_assert(sizeof(Material) == 32, "Material must be 32 bytes for GPU alignment");

/// 48-byte GPU struct
struct Light {
    glm::vec3 direction_or_pos;
    uint32_t type;
    glm::vec3 color;
    float intensity;
    float radius;
    float width;
    float height;
    float angle;
};
static_assert(sizeof(Light) == 48, "Light must be 48 bytes for GPU alignment");

struct Mesh {
    webgpu::Buffer vertex_buffer;
    webgpu::Buffer index_buffer;
    uint32_t index_count;
    std::vector<uint32_t> cpu_indices;
    uint32_t version = 0;
};

struct ObjectSlot {
    uint32_t mesh_index;
    uint32_t material_index{k_no_material};
    glm::mat4 transform;
    std::string prim_path;
    bool active{true};
};

struct LightSlot {
    enum class Type { Distant, Sphere, Rect, Disk, Dome };
    Type type;
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    float intensity{1.0f};
    glm::mat4 transform;
    glm::vec3 direction{0.0f, -1.0f, 0.0f};
    float angle{0.53f};
    float radius{0.0f};
    float width{1.0f};
    float height{1.0f};
    std::string prim_path;
    bool active{true};
    uint32_t version = 0;
};

/// Convert a LightSlot to a GPU-ready Light struct.
Light to_light(const LightSlot& slot);

/// Prim path → slot lookup entry. A single map replaces separate
/// prim_to_object / prim_to_light maps for better cache locality.
struct PrimSlot {
    enum class Kind : uint8_t { Object, Light };
    Kind kind;
    uint32_t index;
};

struct RenderWorld;

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

    uint32_t alloc_object_slot();
    uint32_t alloc_mesh_slot();
    uint32_t alloc_light_slot();
    void free_object_slot(uint32_t i);
    void free_mesh_slot(uint32_t i);
    void free_light_slot(uint32_t i);

    // Mutable accessors for adapter/sync code (friend-gated).
    ObjectSlot& object(uint32_t i);
    Mesh& mesh(uint32_t i);
    LightSlot& light(uint32_t i);
    Material& material(uint32_t i);
    std::vector<Material>& materials();
    std::unordered_map<std::string, uint32_t>& material_cache();
    void set_prim_slot(const std::string& path, PrimSlot slot);
    void mark_light_dirty(uint32_t i);
    void bump_light_version();

   private:
    RenderWorld& m_world;
};

struct RenderWorld {
    // Read-only accessors
    boost::span<const ObjectSlot> get_objects() const;
    boost::span<const Mesh> get_meshes() const;
    boost::span<const LightSlot> get_lights() const;
    boost::span<const Material> get_materials() const;
    uint32_t get_mesh_version() const;
    uint32_t get_light_version() const;
    uint32_t get_material_version() const;

    int find_object_by_prim(std::string_view path) const;
    int find_light_by_prim(std::string_view path) const;

    /// Iterate prim slots without exposing the container.
    /// fn(std::string_view path, PrimSlot slot)
    template <typename F>
    void for_each_prim(F&& fn) const {
        for (const auto& [path, slot] : m_prim_slots) {
            fn(std::string_view{path}, slot);
        }
    }

    // Per-slot dirty tracking
    boost::span<const uint8_t> get_dirty_lights() const;
    void clear_dirty_lights();

    // GPU buffer management
    void prepare_gpu_buffers(const webgpu::Device& device, WGPUQueue queue);
    const webgpu::Buffer& light_buffer() const;
    const webgpu::Buffer& material_buffer() const;
    uint32_t gpu_light_count() const;

    /// Lightweight xform-only update: recomputes world transforms for all
    /// synced prims at or under the given paths. Does not re-upload meshes.
    void update_transforms(const pxr::UsdStageRefPtr& stage,
                           const std::vector<pxr::SdfPath>& dirty_paths);

    /// Begin a batched sync operation. mesh_version is bumped when
    /// the returned scope guard is destroyed. sync_object/remove_prim
    /// calls without a live SyncScope will PRECONDITION-fail.
    [[nodiscard]] SyncScope begin_sync();

    void clear();

   private:
    friend class SyncScope;

    std::vector<Mesh> m_meshes;
    std::vector<ObjectSlot> m_objects;
    std::vector<Material> m_materials;
    std::vector<LightSlot> m_lights;
    std::vector<uint8_t> m_dirty_lights;

    /// Material path → material index (deduplication cache).
    std::unordered_map<std::string, uint32_t> m_material_cache;

    /// Prim path → slot (object or light). Uses std::less<> for transparent
    /// lookup so find() accepts string_view without allocating.
    boost::container::flat_map<std::string, PrimSlot, std::less<>> m_prim_slots;

    uint32_t m_mesh_version = 0;
    uint32_t m_light_version = 0;
    uint32_t m_material_version = 0;

    // GPU buffer state
    webgpu::Buffer m_gpu_light_buffer;
    webgpu::Buffer m_gpu_material_buffer;
    uint32_t m_gpu_light_count = 0;
    uint32_t m_cached_light_version = UINT32_MAX;
    uint32_t m_cached_material_version = UINT32_MAX;

    std::vector<uint32_t> m_free_object_slots;
    std::vector<uint32_t> m_free_mesh_slots;
    std::vector<uint32_t> m_free_light_slots;
};

}  // namespace pts::rendering
