#pragma once

#include <boost/container/flat_map.hpp>

#include <climits>
#include <cstdint>
#include <memory>
#include <string_view>

namespace pts {

namespace webgpu {
class Device;
}

namespace rendering {

class FrameGraph;
struct PassContext;

class IScenePass {
   public:
    virtual ~IScenePass() = default;

    [[nodiscard]] virtual auto name() const noexcept -> std::string_view = 0;
    [[nodiscard]] virtual auto is_ready() const noexcept -> bool = 0;

    virtual void setup(const webgpu::Device& device) = 0;
    virtual void add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) = 0;

   protected:
    /// Lazily create or return cached per-mesh auxiliary data.
    /// Re-creates when mesh_version changes from the cached version.
    /// T may be move-only (e.g. contains webgpu::Buffer).
    template <typename T, typename Factory>
    auto mesh_cache_get(uint32_t mesh_index, uint32_t mesh_version, Factory&& factory) -> T& {
        auto it = m_mesh_cache.find(mesh_index);
        if (it == m_mesh_cache.end()) {
            it = m_mesh_cache.emplace(mesh_index, MeshCacheEntry{}).first;
        }
        auto& entry = it->second;
        if (entry.version != mesh_version || !entry.data) {
            entry.data = ErasedPtr(new T(std::forward<Factory>(factory)()),
                                   [](void* p) { delete static_cast<T*>(p); });
            entry.version = mesh_version;
        }
        return *static_cast<T*>(entry.data.get());
    }

    /// Clear all cached mesh data.
    void mesh_cache_clear() { m_mesh_cache.clear(); }

   private:
    using ErasedPtr = std::unique_ptr<void, void (*)(void*)>;
    struct MeshCacheEntry {
        ErasedPtr data{nullptr, nullptr};
        uint32_t version = UINT32_MAX;
    };
    boost::container::flat_map<uint32_t, MeshCacheEntry> m_mesh_cache;
};

}  // namespace rendering
}  // namespace pts
