#pragma once

#include <core/diagnostics.h>
#include <core/rendering/vertex.h>
#include <core/rendering/webgpu/buffer.h>

#include <boost/container/flat_map.hpp>

#include <climits>
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pts::rendering {

static constexpr uint32_t k_no_material = UINT32_MAX;

struct Material {
    glm::vec3 diffuse_color{1.0f, 1.0f, 1.0f};
    float metallic{0.0f};
    float roughness{0.5f};
    float opacity{1.0f};
    uint32_t _padding[2]{};
};
static_assert(sizeof(Material) == 32, "Material must be 32 bytes for GPU alignment");

struct Mesh {
    webgpu::Buffer vertex_buffer;
    webgpu::Buffer index_buffer;
    uint32_t index_count;
    std::vector<uint32_t> cpu_indices;
};

struct RenderObject {
    uint32_t mesh_index;
    uint32_t material_index{k_no_material};
    glm::mat4 transform;
    std::string prim_path;
    bool active{true};
};

struct Light {
    enum class Type { Distant, Sphere, Rect, Disk, Dome };
    Type type;
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    float intensity{1.0f};
    glm::mat4 transform;
    std::string prim_path;

    // Distant light
    glm::vec3 direction{0.0f, -1.0f, 0.0f};

    // Area/point lights
    float radius{0.0f};
    float width{1.0f};
    float height{1.0f};
    bool active{true};
};

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

   private:
    RenderWorld& m_world;
};

struct RenderWorld {
    std::vector<Mesh> meshes;
    std::vector<RenderObject> objects;
    std::vector<Material> materials;
    std::vector<Light> lights;

    /// Material path → material index (deduplication cache).
    std::unordered_map<std::string, uint32_t> material_cache;

    /// Prim path → slot (object or light). Uses std::less<> for transparent
    /// lookup so find() accepts string_view without allocating.
    boost::container::flat_map<std::string, PrimSlot, std::less<>> prim_slots;

    uint32_t mesh_version = 0;

    /// Begin a batched sync operation. mesh_version is bumped when
    /// the returned scope guard is destroyed. sync_object/remove_prim
    /// calls without a live SyncScope will PRECONDITION-fail.
    [[nodiscard]] SyncScope begin_sync();

    uint32_t alloc_object_slot();
    uint32_t alloc_mesh_slot();
    uint32_t alloc_light_slot();
    void free_object_slot(uint32_t i);
    void free_mesh_slot(uint32_t i);
    void free_light_slot(uint32_t i);
    int find_object_by_prim(std::string_view path) const;
    int find_light_by_prim(std::string_view path) const;
    void clear();

   private:
    friend class SyncScope;
    uint32_t m_sync_depth = 0;
    std::vector<uint32_t> m_free_object_slots;
    std::vector<uint32_t> m_free_mesh_slots;
    std::vector<uint32_t> m_free_light_slots;
};

}  // namespace pts::rendering
