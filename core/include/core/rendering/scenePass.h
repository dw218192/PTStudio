#pragma once

#include <core/diagnostics.h>

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
    /// Lazily create or return cached data keyed by (key, version).
    /// Re-creates when version changes from the cached version.
    /// T may be move-only (e.g. contains webgpu::Buffer).
    ///
    /// If factory is provided, calls it on cache miss to create the entry.
    /// If factory is omitted (nullptr), INVARIANT-fails on cache miss.
    template <typename T, typename Factory>
    auto cache_get(uint32_t key, uint32_t version, Factory&& factory) -> T& {
        auto it = m_cache.find(key);
        if (it == m_cache.end()) {
            it = m_cache.emplace(key, CacheEntry{}).first;
        }
        auto& entry = it->second;
        if (entry.version != version || !entry.data) {
            entry.data = ErasedPtr(new T(std::forward<Factory>(factory)()),
                                   [](void* p) { delete static_cast<T*>(p); });
            entry.version = version;
        }
        return *static_cast<T*>(entry.data.get());
    }

    /// cache_get without factory — asserts that the entry already exists.
    template <typename T>
    auto cache_get(uint32_t key, uint32_t version, std::nullptr_t) -> T& {
        auto it = m_cache.find(key);
        INVARIANT_MSG(it != m_cache.end() && it->second.data && it->second.version == version,
                      "cache miss with no factory");
        return *static_cast<T*>(it->second.data.get());
    }

    /// Clear all cached data.
    void cache_clear() {
        m_cache.clear();
    }

   private:
    using ErasedPtr = std::unique_ptr<void, void (*)(void*)>;
    struct CacheEntry {
        ErasedPtr data{nullptr, nullptr};
        uint32_t version = UINT32_MAX;
    };
    boost::container::flat_map<uint32_t, CacheEntry> m_cache;
};

}  // namespace rendering
}  // namespace pts
