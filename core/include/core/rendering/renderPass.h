#pragma once

#include <core/diagnostics.h>
#include <core/rendering/frameGraph.h>
#include <core/rendering/renderWorld.h>

#include <boost/container/flat_map.hpp>
#include <climits>
#include <cstdint>
#include <glm/mat4x4.hpp>
#include <memory>
#include <string_view>

namespace spdlog {
class logger;
}

namespace pts {

namespace webgpu {
class Device;
}

namespace rendering {

class ShaderLoader;
struct PassContext;

/// Entity type for pass data keying. Determines which entity's
/// version field is used for automatic invalidation.
enum class PassDataKind : uint8_t { Mesh, Light, Material };

class IPass {
   public:
    explicit IPass(const ShaderLoader& shader_loader);
    virtual ~IPass() = default;

    [[nodiscard]] virtual auto name() const noexcept -> std::string_view = 0;
    [[nodiscard]] virtual auto is_ready() const noexcept -> bool = 0;

    /// Initialize the pass. Creates a named logger via LoggingManager (same
    /// sinks/pattern as the rest of the application), computes allowed debug
    /// targets, then calls do_setup().
    void setup(const webgpu::Device& device);

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

    /// A debug-viewable texture produced by this pass.
    struct DebugTarget {
        const char* label;          ///< UI display name (e.g. "Direct Diffuse")
        const char* resource_name;  ///< Frame graph resource name (e.g. "scene_normals")
    };

    /// Debug targets declared by this pass. Returns {pointer to static array, count}.
    /// Stripped to {nullptr, 0} when PTS_DEBUG_VIEWS is not defined.
    [[nodiscard]] virtual auto debug_targets() const noexcept
        -> std::pair<const DebugTarget*, uint32_t> {
        return {nullptr, 0};
    }

    /// Debug targets gated by device limits and build config. Returns empty
    /// when PTS_DEBUG_VIEWS is not defined, stripping all debug target overhead.
    [[nodiscard]] auto effective_debug_targets() const noexcept
        -> std::pair<const DebugTarget*, uint32_t> {
#ifndef PTS_DEBUG_VIEWS
        return {nullptr, 0};
#else
        auto [targets, count] = debug_targets();
        return {targets, std::min(count, m_allowed_debug_count)};
#endif
    }

    [[nodiscard]] auto get_shader_loader() const noexcept -> const ShaderLoader& {
        return *m_shader_loader;
    }

    [[nodiscard]] auto logger() const noexcept -> spdlog::logger& {
        return *m_logger;
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

    /// Frame graph resource helpers — auto-namespace by pass name.
    TextureHandle create_texture(FrameGraph& fg, TextureDesc desc, const char* label = nullptr) {
        return fg.find_or_create(this, desc, label);
    }
    BufferHandle create_buffer(FrameGraph& fg, BufferDesc desc, const char* label = nullptr) {
        return fg.find_or_create_buffer(this, desc, label);
    }
    BufferHandle import_buffer(FrameGraph& fg, WGPUBuffer buf, std::size_t size,
                               const char* label = nullptr) {
        return fg.import_buffer(this, buf, size, label);
    }
    DescriptorHandle create_descriptor(FrameGraph& fg, DescriptorDesc desc,
                                       const char* label = nullptr) {
        return fg.find_or_create_descriptor(this, std::move(desc), label);
    }
    DescriptorBuilder descriptor(FrameGraph& fg, WGPUBindGroupLayout layout,
                                 const char* label = nullptr) {
        return fg.descriptor(this, layout, label);
    }

    /// Lazily create or return per-entity pass data, cached in the world.
    /// Version is read from the entity (Mesh::generation or Light::generation).
    /// Re-creates when the version changes. Cache is destroyed with the world,
    /// so no stale data survives a scene swap.
    template <typename T, typename Factory>
    auto get_or_create_pass_data(PassDataKind kind, uint32_t index, const RenderWorld& world,
                                 Factory&& factory) -> T& {
        auto version = entity_version(kind, index, world);
        auto key = make_key(kind, index);
        auto& map = const_cast<RenderWorld&>(world).pass_data_for(this);
        auto it = map.find(key);
        if (it == map.end()) {
            it = map.emplace(key, RenderWorld::PassDataEntry{}).first;
        }
        auto& entry = it->second;
        if (entry.version != version || !entry.data) {
            entry.data = RenderWorld::ErasedPtr(new T(std::forward<Factory>(factory)()),
                                                [](void* p) { delete static_cast<T*>(p); });
            entry.version = version;
        }
        return *static_cast<T*>(entry.data.get());
    }

    /// get_or_create_pass_data without factory — asserts that the entry already exists.
    template <typename T>
    auto get_or_create_pass_data(PassDataKind kind, uint32_t index, const RenderWorld& world,
                                 std::nullptr_t) -> T& {
        auto version = entity_version(kind, index, world);
        auto key = make_key(kind, index);
        auto& map = const_cast<RenderWorld&>(world).pass_data_for(this);
        auto it = map.find(key);
        INVARIANT_MSG(it != map.end() && it->second.data && it->second.version == version,
                      "pass data miss with no factory");
        return *static_cast<T*>(it->second.data.get());
    }

    /// Per-category pass data — invalidated when *any* entity in the category changes.
    template <typename T, typename Factory>
    auto get_or_create_pass_data(PassDataKind kind, const RenderWorld& world, Factory&& factory)
        -> T& {
        auto version = category_version(kind, world);
        auto key = make_category_key(kind);
        auto& map = const_cast<RenderWorld&>(world).pass_data_for(this);
        auto it = map.find(key);
        if (it == map.end()) {
            it = map.emplace(key, RenderWorld::PassDataEntry{}).first;
        }
        auto& entry = it->second;
        if (entry.version != version || !entry.data) {
            entry.data = RenderWorld::ErasedPtr(new T(std::forward<Factory>(factory)()),
                                                [](void* p) { delete static_cast<T*>(p); });
            entry.version = version;
        }
        return *static_cast<T*>(entry.data.get());
    }

   private:
    const ShaderLoader* m_shader_loader;
    std::shared_ptr<spdlog::logger> m_logger;
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

    static uint64_t make_category_key(PassDataKind kind) {
        return (static_cast<uint64_t>(kind) << 32) | UINT32_MAX;
    }
};

}  // namespace rendering
}  // namespace pts
