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

// --- Slot<T> ---

template <typename T>
class SlotVector;

template <typename T>
class Slot {
   public:
    const T& data() const {
        return m_data;
    }
    const T* operator->() const {
        return &m_data;
    }
    uint32_t generation() const {
        return m_generation;
    }
    bool active() const {
        return m_active;
    }
    const pxr::SdfPath& get_prim_path() const {
        return m_prim_path;
    }

    class WriteGuard {
       public:
        T& operator*() {
            return m_slot->m_data;
        }
        T* operator->() {
            return &m_slot->m_data;
        }
        ~WriteGuard() {
            if (m_slot) ++m_slot->m_generation;
        }
        WriteGuard(const WriteGuard&) = delete;
        WriteGuard& operator=(const WriteGuard&) = delete;
        WriteGuard(WriteGuard&& o) noexcept : m_slot(o.m_slot) {
            o.m_slot = nullptr;
        }
        WriteGuard& operator=(WriteGuard&&) = delete;

       private:
        friend class Slot;
        explicit WriteGuard(Slot& s) : m_slot(&s) {
        }
        Slot* m_slot;
    };

    [[nodiscard]] WriteGuard write() {
        return WriteGuard{*this};
    }
    void activate() {
        m_active = true;
        ++m_generation;
    }
    void deactivate() {
        m_active = false;
        ++m_generation;
    }

   private:
    friend class SlotVector<T>;
    T m_data{};
    pxr::SdfPath m_prim_path;
    uint32_t m_generation = 0;
    bool m_active = false;
};

// --- SlotVector<T> ---

template <typename T>
class SlotVector {
   public:
    uint32_t alloc() {
        uint32_t idx;
        if (!m_free.empty()) {
            idx = m_free.back();
            m_free.pop_back();
            // Reset data to default
            auto w = m_slots[idx].write();
            *w = T{};
        } else {
            m_slots.push_back(Slot<T>{});
            idx = static_cast<uint32_t>(m_slots.size() - 1);
        }
        m_slots[idx].m_prim_path = pxr::SdfPath();
        m_slots[idx].activate();
        return idx;
    }

    void free(uint32_t i) {
        PRECONDITION(i < m_slots.size());
        PRECONDITION(m_slots[i].active());
        m_slots[i].deactivate();
        m_free.push_back(i);
    }

    const Slot<T>& operator[](uint32_t i) const {
        PRECONDITION(i < m_slots.size());
        return m_slots[i];
    }

    typename Slot<T>::WriteGuard write(uint32_t i) {
        PRECONDITION(i < m_slots.size());
        return m_slots[i].write();
    }

    uint32_t size() const {
        return static_cast<uint32_t>(m_slots.size());
    }

    boost::span<const Slot<T>> span() const {
        return {m_slots.data(), m_slots.size()};
    }

    void set_prim_path(uint32_t i, pxr::SdfPath path) {
        PRECONDITION(i < m_slots.size());
        m_slots[i].m_prim_path = std::move(path);
    }

    void clear() {
        m_slots.clear();
        m_free.clear();
    }

   private:
    std::vector<Slot<T>> m_slots;
    std::vector<uint32_t> m_free;
};

// --- Data structs (plain POD, no version/active — those live in Slot<>) ---

struct MeshData {
    webgpu::Buffer vertex_buffer;
    webgpu::Buffer index_buffer;
    uint32_t index_count = 0;
    std::vector<uint32_t> cpu_indices;
    std::vector<Vertex> cpu_vertices;
};

struct ObjectData {
    uint32_t mesh_index = 0;
    uint32_t material_index{k_no_material};
    glm::mat4 transform{1.0f};
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

/// Prim path → slot lookup entry. A single map replaces separate
/// prim_to_object / prim_to_light maps for better cache locality.
struct PrimSlot {
    enum class Kind : uint8_t { Object, Light, Camera };
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
    uint32_t alloc_camera_slot();
    void free_object_slot(uint32_t i);
    void free_mesh_slot(uint32_t i);
    void free_light_slot(uint32_t i);
    void free_camera_slot(uint32_t i);

    // Write guards for adapter/sync code.
    Slot<ObjectData>::WriteGuard write_object(uint32_t i);
    Slot<MeshData>::WriteGuard write_mesh(uint32_t i);
    Slot<LightData>::WriteGuard write_light(uint32_t i);
    Slot<CameraData>::WriteGuard write_camera(uint32_t i);

    // Read-only accessors through scope (for prim_path lookup etc.)
    const Slot<ObjectData>& object(uint32_t i) const;
    const Slot<MeshData>& mesh(uint32_t i) const;
    const Slot<LightData>& light(uint32_t i) const;
    const Slot<CameraData>& camera(uint32_t i) const;

    Material& material(uint32_t i);
    std::vector<Material>& materials();
    std::unordered_map<std::string, uint32_t>& material_cache();
    void set_prim_path(uint32_t slot_index, PrimSlot::Kind kind, pxr::SdfPath path);

   private:
    RenderWorld& m_world;
};

struct RenderWorld {
    // Read-only accessors
    boost::span<const Slot<ObjectData>> get_objects() const;
    boost::span<const Slot<MeshData>> get_meshes() const;
    boost::span<const Slot<LightData>> get_lights() const;
    boost::span<const Slot<CameraData>> get_cameras() const;
    boost::span<const Material> get_materials() const;
    uint32_t get_mesh_version() const;
    uint32_t get_light_version() const;
    uint32_t get_material_version() const;

    int find_object_by_prim(const pxr::SdfPath& path) const;
    int find_light_by_prim(const pxr::SdfPath& path) const;
    int find_camera_by_prim(const pxr::SdfPath& path) const;

    /// Iterate prim slots without exposing the container.
    /// fn(const pxr::SdfPath& path, PrimSlot slot)
    template <typename F>
    void for_each_prim(F&& fn) const {
        for (const auto& [path, slot] : m_prim_slots) {
            fn(path, slot);
        }
    }

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

    /// Upload GPU buffers for all meshes that have CPU vertex data.
    /// Call on the main thread after building the RenderWorld off-thread.
    void upload_all_meshes(const webgpu::Device& device);

    void clear();

   private:
    friend class SyncScope;

    SlotVector<MeshData> m_meshes;
    SlotVector<ObjectData> m_objects;
    std::vector<Material> m_materials;
    SlotVector<LightData> m_lights;
    SlotVector<CameraData> m_cameras;

    /// Material path → material index (deduplication cache).
    std::unordered_map<std::string, uint32_t> m_material_cache;

    /// Prim path → slot (object or light). SdfPath has operator< and O(1)
    /// equality via interned strings.
    boost::container::flat_map<pxr::SdfPath, PrimSlot> m_prim_slots;

    uint32_t m_mesh_version = 0;
    uint32_t m_light_version = 0;
    uint32_t m_material_version = 0;

    // GPU buffer state
    webgpu::Buffer m_gpu_light_buffer;
    webgpu::Buffer m_gpu_material_buffer;
    uint32_t m_gpu_light_count = 0;
    uint32_t m_cached_light_version = UINT32_MAX;
    uint32_t m_cached_material_version = UINT32_MAX;

    // Per-slot generation cache for partial light updates
    std::vector<uint32_t> m_cached_light_generations;
};

}  // namespace pts::rendering
