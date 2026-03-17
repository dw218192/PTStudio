# Shader Hot-Reload + BRDF Lobe Panel

## Context

Shader iteration is slow: edit `.slang` → prebuild → rebuild C++ → relaunch. This plan adds two features to cut that loop:

1. **Shader hot-reload** (core) — in native debug builds, load `.wgsl` from disk with an explicit reload button. Release/Emscripten keeps embedded resources.
2. **BRDF lobe panel** (editor) — ImGui panel rendering a GGX lobe as a displaced sphere into an offscreen texture, displayed via `ImGui::Image`.

## Ticket 1: `shader-hot-reload` — Core shader loader with hot-reload

### What
A `ShaderLoader` utility in core that abstracts shader loading. In debug native builds (`PTS_SHADER_HOT_RELOAD`), it reads `.wgsl` from disk on explicit `reload_all()`. Otherwise, it delegates to embedded resources. No file watching — reload is user-triggered via a button.

**Import-aware reloading**: When a shader is reloaded, all shaders that import it must also be recompiled. Since slangc handles `.slang` → `.wgsl` compilation (resolving imports), the reload flow is:
1. User clicks "Reload Shaders" button
2. ShaderLoader invokes `slangc` (via `repo slangc` or direct subprocess) to recompile all registered `.slang` sources — this handles imports automatically
3. ShaderLoader re-reads the output `.wgsl` files from disk
4. Passes rebuild their pipelines from the new `.wgsl`

This means the loader tracks `.slang` source paths (for recompilation) and `.wgsl` output paths (for loading).

### Files to create
- `core/include/core/rendering/shaderLoader.h`
- `core/src/rendering/shaderLoader.cpp`

### Files to modify
- `CMakeLists.txt` — add `PTS_SHADER_HOT_RELOAD` and `PTS_WORKSPACE_ROOT` to `developer_flags`, following the `TRACY_ENABLE` pattern:
  ```cmake
  $<$<AND:$<NOT:$<BOOL:${EMSCRIPTEN}>>,$<CONFIG:Debug>>:PTS_SHADER_HOT_RELOAD>
  $<$<AND:$<NOT:$<BOOL:${EMSCRIPTEN}>>,$<CONFIG:Debug>>:PTS_WORKSPACE_ROOT="${CMAKE_SOURCE_DIR}">
  ```
- `core/CMakeLists.txt` — add `shaderLoader.cpp` to sources
- `core/include/core/rendering/scenePass.h` — add `virtual void on_shaders_reloaded(const webgpu::Device&) {}` (default no-op)

### API

```cpp
namespace pts::rendering {

// Function pointer matching the generated get_resource() signature
using EmbeddedGetter = std::optional<std::string_view>(*)(std::string_view);

class ShaderLoader {
public:
    explicit ShaderLoader(std::shared_ptr<spdlog::logger> logger);

    // Register a shader. resource_key is the embedded resource lookup key.
    // disk_path is the .wgsl path relative to workspace root.
    // embedded_getter is the namespace::get_resource function pointer.
    void register_shader(std::string_view resource_key,
                         std::string_view disk_path,
                         EmbeddedGetter embedded_getter);

    // Load shader source. Hot-reload: reads from disk. Otherwise: embedded.
    // Returns nullopt on disk read failure (caller keeps last-good).
    [[nodiscard]] auto load(std::string_view resource_key) const
        -> std::optional<std::string>;

    // Recompile all shaders (runs slangc) and re-read .wgsl from disk.
    // Returns true if any shader changed. No-op in non-hot-reload builds.
    [[nodiscard]] auto reload_all() -> bool;
};

}
```

### Implementation notes
- `#ifdef PTS_SHADER_HOT_RELOAD`: `reload_all()` runs `repo slangc` as a subprocess (or invokes slangc directly), then re-reads all registered `.wgsl` files from disk. Returns true if any content differs from cached.
- `#else`: `reload_all()` returns false, `load()` calls `embedded_getter(resource_key)`
- No file watching, no mtime tracking — reload is explicit
- Log recompilation results and errors

## Ticket 2: `editor-hot-reload` — Wire hot-reload into editor passes

### What
Integrate `ShaderLoader` into `EditorApplication`. Register all editor shaders. Add a "Reload Shaders" button (visible only in hot-reload builds). ForwardPass adopts `on_shaders_reloaded` to rebuild its pipeline.

### Files to modify
- `editor/src/include/editorApplication.h` — add `ShaderLoader m_shader_loader` member
- `editor/src/editorApplication.cpp` — in `on_ready()`: register editor shaders. Add "Reload Shaders" button (e.g. in menu bar or debug panel, guarded by `#ifdef PTS_SHADER_HOT_RELOAD`). On click: call `m_shader_loader.reload_all()`, then dispatch `on_shaders_reloaded` to all passes.
- `editor/src/passes/forwardPass.h` — override `on_shaders_reloaded`
- `editor/src/passes/forwardPass.cpp` — implement pipeline rebuild

### Pipeline rebuild strategy
When `on_shaders_reloaded` fires:
1. `m_shader_loader->load(key)` → new WGSL source
2. `device.create_shader_module_from_source()` — if it throws (compile error), catch, log, return
3. Rebuild `RenderPipeline` via `RenderPipelineBuilder` with same config
4. Move-assign new shader + pipeline into `Ready` struct (old ones released by RAII)
5. Bind group layout unchanged (shader interface hasn't changed) — no bind group rebuild needed

Other passes (Grid, Wireframe, Picking) can adopt incrementally later.

## Ticket 3: `lobe-panel-shader` — Lobe shader + config

### What
Write `editor/shaders/lobe.slang` and add it to the prebuild pipeline.

### Files to create
- `editor/shaders/lobe.slang`

### Files to modify
- `config.yaml` — add slangc, shader_codegen, embed entries for lobe shader

### Shader design
Imports `lighting.slang` from `core/shaders/` (already in slangc search paths).

```
Uniforms: mat4 mvp, float3 light_dir, float roughness, float metallic, float scale, pad

No vertex buffer — vertices generated from vertex_id.

Vertex shader:
  - Derive (theta, phi) grid coords from vertex_id and grid dimensions (e.g. 128×64)
  - direction = spherical_to_cartesian(theta, phi) on upper hemisphere
  - N = float3(0,0,1) (surface normal, fixed)
  - V = direction (outgoing direction ω_o)
  - L = light_dir uniform (incoming light, 2 DOF: azimuth + elevation)
  - Evaluate combined Cook-Torrance specular via D_GGX * G_SmithSchlick * F_Schlick
  - Displace: out_pos = direction * (base_radius + magnitude * scale)
  - Transform by MVP, pass magnitude as varying

Fragment shader:
  - Heat map color from magnitude varying (blue→cyan→green→yellow→red)

Draw call: use index buffer for triangle grid, or draw (cols-1)*(rows-1)*6
vertices with indices computed from vertex_id.
```

### config.yaml additions
- `slangc.shaders`: `editor/shaders/lobe.slang` → `editor/generated/shaders/lobe.wgsl` (reflect: true)
- `shader_codegen.shaders`: reflect JSON → `editor/generated/lobe_shader_metadata.h`, namespace `editor_lobe_shader`
- `embed.resources[0].input`: add `editor/generated/shaders/lobe.wgsl` to editor embed group

## Ticket 4: `lobe-panel` — LobePass + editor integration

### What
Implement `LobePass` as a new editor pass. Add BRDF Lobe ImGui panel to the editor.

### Files to create
- `editor/src/passes/lobePass.h`
- `editor/src/passes/lobePass.cpp`

### Files to modify
- `editor/src/include/editorApplication.h` — add `draw_brdf_panel()` method, lobe texture ref member
- `editor/src/editorApplication.cpp` — instantiate LobePass, add to passes, call `draw_brdf_panel()` in render, handle offscreen texture display
- `editor/CMakeLists.txt` — add lobePass.cpp to sources

### LobePass design
Follows `GridPass` pattern (simplest pass — no per-object data, one draw call).

```cpp
class LobePass final : public rendering::IScenePass {
    struct Ready {
        webgpu::ShaderModule shader;
        webgpu::RenderPipeline lobe_pipeline;     // triangle topology, no vertex buffer
        webgpu::RenderPipeline ref_pipeline;       // line topology (reference geo)
        webgpu::Buffer uniform_buffer;
        webgpu::Buffer ref_vb;                      // reference lines (CPU-generated)
        WGPUBindGroup bind_group;
        WGPUBindGroupLayout bind_group_layout;
    };
    // ImGui params
    float m_roughness = 0.5f, m_metallic = 0.0f;
    float m_light_azimuth = 0.0f, m_light_elevation = 45.0f;  // 2 DOF

    float m_scale = 1.0f;
    bool m_show_reference = true;
};
```

- `setup()`: create reference line geometry (hemisphere wireframe + axis lines), create shader/pipeline/buffers. No sphere mesh — lobe vertices generated from vertex_id in the shader. Draw call: `draw(vertex_count)` where `vertex_count = (cols-1)*(rows-1)*6` (two triangles per grid cell).
- `add_to_frame_graph()`: create `"lobe_color"` resource (256×256), render lobe + reference geo into it
- Separate `draw_imgui_controls()` method for the panel UI
- The pass does NOT use `PassContext` camera — it has its own orbit camera or fixed view. Simplest: fixed view looking at origin, user rotates light direction instead.

### Editor integration
- `draw_brdf_panel()`: ImGui window "BRDF Lobe", contains:
  - **Light direction widget**: a 2D circle widget where the user clicks/drags to set azimuth and elevation (2 DOF). The position within the circle maps to the hemisphere of incoming light directions. Normal is fixed at (0,0,1).
  - Roughness slider [0,1], metallic slider [0,1], scale slider
  - Show Reference checkbox
  - `ImGui::Image` of lobe texture below controls
- Follow existing `draw_scene_viewport()` pattern for `ImGui::Image` with `TextureRef`
- Add to docking layout as an optional panel

### Reference geometry
Rendered in same pass with a second pipeline (line topology):
- Wireframe hemisphere (16 lat × 16 lon lines)
- RGB axes (X=red, Y=green, Z=blue, length ~1.2)
- Yellow arrow for light direction
- Depth test on, depth write off (always visible through lobe)

## Ticket order (dependencies)

1. `shader-hot-reload` — no deps
2. `lobe-panel-shader` — no deps (can parallel with 1)
3. `editor-hot-reload` — depends on 1
4. `lobe-panel` — depends on 2, benefits from 1 (hot-reload during development)

## Verification

1. **Hot-reload**: build native Debug, launch editor, edit `editor/shaders/forward.slang` on disk, click "Reload Shaders", confirm viewport updates without restart
2. **Hot-reload with imports**: edit `core/shaders/lighting.slang`, click "Reload Shaders", confirm all dependent shaders (forward, lobe) update
3. **Hot-reload error handling**: introduce syntax error in `.slang`, click reload, confirm editor logs error and keeps rendering with last-good shader
3. **Lobe panel**: open BRDF Lobe panel in editor, verify sphere deforms with roughness/metallic changes, reference geometry toggles
4. **Build**: `repo build` succeeds for native (Debug+Release) and emscripten
5. **Tests**: `repo test` passes — add unit test for `ShaderLoader` embedded fallback path
