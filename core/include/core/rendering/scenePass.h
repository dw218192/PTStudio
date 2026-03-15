#pragma once

#include <any>
#include <climits>
#include <cstdint>
#include <string_view>
#include <unordered_map>

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
    /// Factory receives no arguments — the caller captures what it needs.
    template <typename T, typename Factory>
    auto mesh_cache_get(uint32_t mesh_index, uint32_t mesh_version, Factory&& factory) -> T& {
        auto& entry = m_mesh_cache[mesh_index];
        if (entry.version != mesh_version || !entry.data.has_value()) {
            entry.data = std::forward<Factory>(factory)();
            entry.version = mesh_version;
        }
        return std::any_cast<T&>(entry.data);
    }

    /// Clear all cached mesh data.
    void mesh_cache_clear() { m_mesh_cache.clear(); }

   private:
    struct MeshCacheEntry {
        std::any data;
        uint32_t version = UINT32_MAX;
    };
    std::unordered_map<uint32_t, MeshCacheEntry> m_mesh_cache;
};

}  // namespace rendering
}  // namespace pts
