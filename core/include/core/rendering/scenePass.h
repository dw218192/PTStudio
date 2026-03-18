#pragma once

#include <core/diagnostics.h>
#include <core/rendering/renderWorld.h>

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

/// Entity type for pass data keying. Determines which entity's
/// version field is used for automatic invalidation.
enum class PassDataKind : uint8_t { Mesh, Light };

class IScenePass {
   public:
    virtual ~IScenePass() = default;

    [[nodiscard]] virtual auto name() const noexcept -> std::string_view = 0;
    [[nodiscard]] virtual auto is_ready() const noexcept -> bool = 0;

    virtual void setup(const webgpu::Device& device) = 0;
    virtual void add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) = 0;

    /// Called when shaders have been hot-reloaded. Override to rebuild pipelines.
    virtual void on_shaders_reloaded(const webgpu::Device& device) {
    }

   protected:
    /// Lazily create or return per-entity pass data.
    /// Version is read automatically from the entity (Mesh::version or Light::version).
    /// Re-creates when the entity's version changes from the cached version.
    /// T may be move-only (e.g. contains webgpu::Buffer).
    template <typename T, typename Factory>
    auto get_or_create_pass_data(PassDataKind kind, uint32_t index, const RenderWorld& world,
                                 Factory&& factory) -> T& {
        auto version = entity_version(kind, index, world);
        auto key = make_key(kind, index);
        auto it = m_pass_data.find(key);
        if (it == m_pass_data.end()) {
            it = m_pass_data.emplace(key, PassDataEntry{}).first;
        }
        auto& entry = it->second;
        if (entry.version != version || !entry.data) {
            entry.data = ErasedPtr(new T(std::forward<Factory>(factory)()),
                                   [](void* p) { delete static_cast<T*>(p); });
            entry.version = version;
        }
        return *static_cast<T*>(entry.data.get());
    }

    /// get_or_create_pass_data without factory — asserts that the entry already exists
    /// and its version matches.
    template <typename T>
    auto get_or_create_pass_data(PassDataKind kind, uint32_t index, const RenderWorld& world,
                                 std::nullptr_t) -> T& {
        auto version = entity_version(kind, index, world);
        auto key = make_key(kind, index);
        auto it = m_pass_data.find(key);
        INVARIANT_MSG(it != m_pass_data.end() && it->second.data && it->second.version == version,
                      "pass data miss with no factory");
        return *static_cast<T*>(it->second.data.get());
    }

    /// Clear all pass data.
    void clear_pass_data() {
        m_pass_data.clear();
    }

   private:
    static uint32_t entity_version(PassDataKind kind, uint32_t index, const RenderWorld& world) {
        switch (kind) {
            case PassDataKind::Mesh:
                return world.get_meshes()[index].version;
            case PassDataKind::Light:
                return world.get_lights()[index].version;
        }
        INVARIANT_MSG(false, "unknown PassDataKind");
    }

    static uint64_t make_key(PassDataKind kind, uint32_t index) {
        return (static_cast<uint64_t>(kind) << 32) | index;
    }

    using ErasedPtr = std::unique_ptr<void, void (*)(void*)>;
    struct PassDataEntry {
        ErasedPtr data{nullptr, nullptr};
        uint32_t version = UINT32_MAX;
    };
    boost::container::flat_map<uint64_t, PassDataEntry> m_pass_data;
};

}  // namespace rendering
}  // namespace pts
