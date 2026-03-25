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
class ShaderLoader;
struct PassContext;

/// Entity type for pass data keying. Determines which entity's
/// version field is used for automatic invalidation.
enum class PassDataKind : uint8_t { Mesh, Light, Material };

class IRenderPass {
   public:
    explicit IRenderPass(const ShaderLoader& shader_loader) : m_shader_loader(&shader_loader) {
    }
    virtual ~IRenderPass() = default;

    [[nodiscard]] virtual auto name() const noexcept -> std::string_view = 0;
    [[nodiscard]] virtual auto is_ready() const noexcept -> bool = 0;

    /// Initialize the pass. Computes allowed debug targets, then calls do_setup().
    void setup(const webgpu::Device& device) {
        compute_allowed_debug_targets(device);
        do_setup(device);
    }

    virtual void add_to_frame_graph(FrameGraph& fg, const PassContext& ctx) = 0;

    /// Called when shaders have been hot-reloaded. Default re-runs setup().
    virtual void on_shaders_reloaded(const webgpu::Device& device) {
        setup(device);
    }

    /// Draw pass-specific ImGui windows/controls. Called during the UI phase.
    virtual void draw_imgui() {
    }

    /// Draw inline controls in the viewport menu bar. Called between BeginMenuBar/EndMenuBar.
    virtual void draw_viewport_controls() {
    }

    /// Draw debug overlays on the viewport's ImGui draw list.
    /// Called after the viewport image, with viewport screen-space params.
    struct ViewportOverlayParams {
        glm::mat4 view_proj;
        float x, y, w, h;
    };
    virtual void draw_viewport_overlay(const ViewportOverlayParams& params) {
    }

    /// Cache texture refs after frame graph execute, for ImGui display next frame.
    virtual void update_texture_refs(FrameGraph& fg) {
    }

    /// Whether this pass requires the scene viewport to render.
    [[nodiscard]] virtual auto requires_viewport() const noexcept -> bool {
        return true;
    }

    /// Debug target names declared by this pass. Each name corresponds to an
    /// MRT color attachment (SV_Target1..N) that the shader writes every frame.
    /// Returns {pointer to static array, count}. Default: no debug targets.
    [[nodiscard]] virtual auto debug_target_names() const noexcept
        -> std::pair<const char* const*, uint32_t> {
        return {nullptr, 0};
    }

    /// Debug targets gated by device limits. Returns the same names array but
    /// with count capped to what the device can actually support. Only valid
    /// after setup() has been called.
    [[nodiscard]] auto effective_debug_target_names() const noexcept
        -> std::pair<const char* const*, uint32_t> {
        auto [names, count] = debug_target_names();
        return {names, std::min(count, m_allowed_debug_count)};
    }

    [[nodiscard]] auto get_shader_loader() const noexcept -> const ShaderLoader& {
        return *m_shader_loader;
    }

    /// Load the pass shader, automatically selecting the no-debug-targets
    /// variant when the device limit requires it. Shaders that declare debug
    /// MRT outputs must guard them with `#ifndef NO_DEBUG_TARGETS`.
    /// The variant is loaded from an embedded resource whose key is derived
    /// by inserting "_no_debug" before the extension (e.g. forward.wgsl →
    /// forward_no_debug.wgsl).
    [[nodiscard]] auto load_pass_shader(std::string_view resource_key) const -> std::string;

   protected:
    virtual void do_setup(const webgpu::Device& device) = 0;
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

    /// Per-category pass data — cached result invalidated when *any* entity in
    /// the category changes.  Same API as the per-entity variant but without an
    /// index: the key is the kind alone.
    template <typename T, typename Factory>
    auto get_or_create_pass_data(PassDataKind kind, const RenderWorld& world, Factory&& factory)
        -> T& {
        auto version = category_version(kind, world);
        auto key = make_category_key(kind);
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

    /// Clear all pass data.
    void clear_pass_data() {
        m_pass_data.clear();
    }

   private:
    const ShaderLoader* m_shader_loader;
    uint32_t m_allowed_debug_count = UINT32_MAX;

    void compute_allowed_debug_targets(const webgpu::Device& device);

    static uint32_t entity_version(PassDataKind kind, uint32_t index, const RenderWorld& world) {
        switch (kind) {
            case PassDataKind::Mesh:
                return world.get_meshes()[index].generation();
            case PassDataKind::Light:
                return world.get_lights()[index].generation();
            case PassDataKind::Material:
                break;
        }
        INVARIANT_MSG(false, "per-entity version not supported for this PassDataKind");
    }

    static uint32_t category_version(PassDataKind kind, const RenderWorld& world) {
        switch (kind) {
            case PassDataKind::Mesh:
                return world.get_mesh_version();
            case PassDataKind::Light:
                return world.get_light_version();
            case PassDataKind::Material:
                return world.get_material_version();
        }
        INVARIANT_MSG(false, "unknown PassDataKind");
    }

    static uint64_t make_key(PassDataKind kind, uint32_t index) {
        return (static_cast<uint64_t>(kind) << 32) | index;
    }

    /// Category-wide key uses a sentinel index that can't collide with entity keys.
    static uint64_t make_category_key(PassDataKind kind) {
        return (static_cast<uint64_t>(kind) << 32) | UINT32_MAX;
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
