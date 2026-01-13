That’s a good ordering. If you do **DLL plugins first**, you lock the ABI seam early and everything else (rendergraph/renderworld/etc.) can evolve *behind* the seam without touching the editor again.

Below is a **minimal-refactor plugin architecture** + **milestones** to get “legacy Vulkan renderer running as a plugin” quickly, without committing to a full RHI/rendergraph yet.

---

## The plugin system you want (upfront)

### Principles

* **C ABI at the boundary** (stable across compilers/settings; no C++ ABI pain).
* **Plugin owns its memory** (host never `delete`s plugin objects; host calls plugin’s `destroy()`).
* **Start with “Renderer Plugin” as a black box** (it can own Vulkan device/swapchain for now).

  * Later you can split into `RHI backend plugin` + `Pipeline plugins` without changing the loader.

---

## `pts_plugin_api` (one header, shared by host + plugins)

### One required exported symbol per DLL

`pts_plugin_get_desc`

It returns a static descriptor struct (or fills one) containing:

* `api_version` (host ↔ plugin compatibility)
* `plugin_id` / `display_name`
* `plugin_kind` (renderer / rhi / pipeline)
* function pointers: `create()`, `destroy()`, etc.

### Host services struct

Host passes a `PtsHostServices` table into `create()`:

* logging callback
* file read callback
* high-res timer
* “window handle” (Win32 HWND or abstract handle)
* optional: paths (asset root, shader cache)
* optional: “GPU capture” hooks (begin/end marker) if you want later

**Important**: don’t pass STL strings/vectors across the ABI. Use `const char*`, `uint32_t`, and spans.

---

## Renderer plugin ABI (v0: black box renderer)

Start with a single plugin kind: **Renderer**.

### Renderer instance vtable-style

Plugin returns a `PtsRenderer*` opaque pointer plus a function table:

Required functions:

* `on_load(PtsHostServices*)` (optional, once)
* `create_renderer(desc, out_renderer)`
* `destroy_renderer(renderer)`
* `resize(renderer, w, h)`
* `render(renderer, FrameParams*)`
* `shutdown(renderer)`

Optional:

* `get_caps(renderer)` (e.g., “ray tracing supported”, “requires Vulkan”)
* `get_debug_outputs(renderer, out_list)` (names + handles)
* `get_settings_schema(renderer, out_props)` and `set_setting(...)` OR a serialized blob

### UI integration (avoid ImGui in plugins initially)

Do **not** compile ImGui into plugins. It’s the fastest way to ODR/global-state misery.

Instead:

* plugin exposes **settings + debug outputs** (as POD)
* host/editor draws the UI with ImGui and just calls plugin setters

This keeps the plugin boundary clean and avoids huge refactors.

---

## Directory / build layout (simple)

* `pts_plugin_api/` (header-only)
* `ptstudio_editor/` (exe, contains PluginManager)
* `plugins/`

  * `renderer_vk_legacy/` (dll)
  * `renderer_gl_legacy/` (dll later)

---

## Milestones (what to build, in order)

### Milestone P0 — Loader + registry in the editor

**Deliverables**

* `PluginManager`:

  * scans `./plugins/*.dll` (or `./plugins/renderers/*.dll`)
  * `LoadLibrary` + `GetProcAddress("pts_plugin_get_desc")`
  * checks `api_version`
  * lists available plugins
* CLI/config selection: `--renderer=<plugin_id>`

**Exit**

* Editor can enumerate plugins and choose one (even if it doesn’t render yet).

---

### Milestone P1 — Define `Renderer` plugin ABI (v0)

**Deliverables**

* `pts_plugin_api.h` with:

  * versioning
  * descriptor structs
  * host services table
  * renderer function table

**Exit**

* You can write a “NullRenderer” plugin that clears screen / logs calls.

---

### Milestone P2 — Convert legacy Vulkan to `renderer_vk_legacy.dll` (minimal refactor)

**Approach to minimize churn**

* Keep `vulkan_raytracer` mostly intact.
* Build `renderer_vk_legacy.dll` that **links against the existing static lib** and wraps it with the plugin ABI.
* The wrapper handles:

  * window handle / swapchain creation
  * calling your existing `init/render/resize/shutdown` functions

**Exit**

* `ptstudio_editor --renderer=vk_legacy` boots and renders your existing output.

---

### Milestone P3 — Settings & debug output bridge (still no rendergraph/world)

**Deliverables**

* `get_debug_outputs()` returns e.g.:

  * “Final”
  * “Albedo”
  * “Normals”
  * “Depth”
  * etc.
* `get_settings_schema()` returns a list of properties:

  * name, type (bool/int/float), ptr-or-id, min/max, step
* Host draws an ImGui panel from that schema and calls `set_setting()`

**Exit**

* You can tweak legacy Vulkan parameters from editor UI without the plugin knowing about ImGui.

---

### Milestone P4 — Hot reload (optional, but doable early)

Start conservative:

* reload requires “stop rendering → destroy renderer → unload DLL → load DLL → create renderer”
* do **not** unload while GPU work is in flight
* or simplest: “reload = restart editor” until stability

**Exit**

* You can iterate on the Vulkan plugin without relinking the editor.

---

## The critical pitfalls (so you don’t step on landmines)

1. **C++ ABI across DLL boundary**

   * Don’t export C++ classes as your public interface.
   * Don’t pass STL types across.

2. **Memory ownership**

   * Host never frees plugin memory.
   * Plugin provides `destroy_*` for everything it allocates.

3. **ImGui**

   * Don’t compile ImGui into plugins initially.
   * Keep UI in host via settings schema.

4. **Unloading**

   * Start by **never unloading** plugins (load once, keep loaded).
   * Add unload only after you’re confident about resource teardown.

5. **CRT / runtime mismatch (Windows)**

   * Ensure editor + plugins use the same runtime (/MD) and compiler settings to avoid subtle issues.

---

## Why this ordering is great

Once `renderer_vk_legacy.dll` runs, you can start “replacing the inside” (rendergraph, renderworld, pipeline split) *without* touching:

* the editor,
* the loader,
* the public ABI.

That’s the best way to avoid refactors between milestones.

---

If you want to align this to your current code quickly: I only still have that one renderer header you attached earlier; I got a system note that some other previously uploaded files expired. If you re-upload your current `vulkan_raytracer` public entrypoints (whatever `init/render/shutdown` looks like), I can map *exactly* how the wrapper DLL should call into it with almost no edits.
