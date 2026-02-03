# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PTStudio is a modular 3D scene editor and renderer written in C++17. It uses WebGPU for cross-platform rendering (desktop via Dawn, web via Emscripten), OpenUSD for scene description, and a plugin system for extensible renderer implementations.

## Build System

### Tooling Architecture
- **Conan** for dependency management (see `conanfile.py`)
- **CMake** for build system generation
- **Python-based tooling**: All build tools are Python packages in `tools/` directory
- **Hermetic environment**: Tools are auto-bootstrapped into `_tools/venv/`

### Common Commands

**Note:** Use `.\pts.cmd` on Windows (cmd/PowerShell) or `./pts` on bash/Unix-like systems.

Build the project:
```bash
.\pts.cmd build
# or on bash:
./pts build
```

Build options:
- `-x, --rebuild`: Clean rebuild (removes build folder)
- `-u, --update-lock`: Force regeneration of `conan.lock`
- `-c, --configure-only`: Run Conan install and CMake configure, skip build
- `-b, --build-only`: Skip Conan/CMake configure, only build
- `--build-type {Debug,Release,RelWithDebInfo,MinSizeRel}`: Set build configuration (default: Debug)
- `--conan-profile PROFILE`: Specify Conan profile
- `--platform {wasm,windows-x64,linux-x64,...}`: Target platform (auto-detected by default)

Run tests:
```bash
.\pts.cmd test
# or: ./pts test
```

Format code:
```bash
.\pts.cmd format
# or: ./pts format
```

See all available commands:
```bash
.\pts.cmd --help
# or: ./pts --help
```

### Build Output Structure
- Build artifacts: `_build/<configuration>/`
- Executables: `_build/<configuration>/bin/`
- Plugins (dynamic): `_build/<configuration>/bin/plugins/`
- Logs: `_logs/`

## Code Architecture

### Module Structure

**core/** - Static library providing foundation functionality
- Public API in `core/include/` and `core/api/` (ABI-stable C API for plugins)
- Windowing abstraction (`IWindowing`, `IViewport`) with GLFW/null/Emscripten backends
- WebGPU rendering (`WebGpuContext`, `Device`, `Surface`, `Pipeline`, `Buffer`, `Texture`)
- Render graph system (`IRenderGraph` internal, `PtsRenderGraphApi` C API)
- Plugin system (`PluginManager`, C ABI-stable plugin interface)
- ImGui integration (`IImguiWindowing`, `IImguiRendering`)

**editor/** - Main executable
- Extends `GUIApplication` with ImGui-based 3D editor UI
- Panels: Scene Settings, Inspector, Scene Viewport, Console
- Hosts renderer plugins via `RendererPluginInterfaceV1`

**plugins/** - Dynamic (or static on WASM) plugin modules
- `renderers/editor/`: Example renderer demonstrating render graph usage
- `test_plugin/`: Reference plugin showing interface patterns
- Built with `pts_add_plugin()` CMake function from `cmake/PluginHelpers.cmake`

**hello_triangle/** - Simple WebGPU example (learning/testing)

### Rendering System

**WebGPU Initialization:**
The `WebGpuContext` uses a state machine pattern to handle async GPU initialization:
- `InitializingState`: Adapter/device request in flight
- `ReadyState`: Device and surface are valid and usable
- `FailedState`: Initialization failed

Applications poll `tick_init()` during startup to advance the state machine without blocking the event loop.

**Render Graph API:**
The render graph uses a frame-based builder pattern exposed through `PtsRenderGraphApi`:
1. `begin()` - Start frame graph construction
2. Create resources: `create_texture()`, `create_buffer()`, `create_sampler()`
3. Import external resources: `import_texture()`, `import_buffer()`
4. Add passes: `add_pass()` with `PtsPassDesc` (dependencies + encode callback)
5. `end()` - Compile and execute graph

Resources can be:
- **Transient**: Valid only within current frame (between `begin()`/`end()`)
- **Persistent**: Survive across frames (host-owned)
- **Imported**: Externally owned, graph just references

The **blackboard** (`get_blackboard()`) provides key-value storage for sharing handles between passes.

**Application Lifecycle:**
```
Application::Application()
  └─ Create windowing, viewport, WebGpuContext (async init starts)

Application::run()
  └─ Poll tick_init() until WebGPU ready
  └─ Loop: run_one_frame() (pump events → loop(dt) → present)

GUIApplication adds:
  └─ ImGui integration (new frame → render → draw to texture)
  └─ Render graph wrapper exposed to plugins
```

### Plugin System

**Architecture:**
- Plugins use **C ABI** for cross-compiler compatibility (headers in `core/api/core/plugin.h`)
- Dynamic loading via Boost.DLL on desktop, static linking on Emscripten
- Two plugin kinds: `PTS_PLUGIN_KIND_SUBSYSTEM`, `PTS_PLUGIN_KIND_RENDERER`

**Plugin Lifecycle:**
```
PtsPluginDescriptor (static metadata)
  ├─ create() → PluginHandle (opaque C++ instance)
  ├─ on_load() → bool (initialization)
  ├─ [runtime] query_interface(handle, iid) → void* (function table)
  ├─ on_unload() (cleanup)
  └─ destroy(handle)
```

**Creating a Plugin:**
1. Define interface struct with function pointers (C-compatible POD)
2. Use metaprogramming helpers in `core/api/core/plugin.h`:
   - `PTS_INTERFACE_ID("interface.name.v1")` - Declare interface ID
   - `PTS_INTERFACE(InterfaceStruct)` - Register in type list
   - `PTS_METHOD(method_name)` - Generate wrapper functions
   - `PTS_PLUGIN_INTERFACES(InterfaceList...)` - Build compile-time registry
   - `PTS_PLUGIN_DEFINE(...)` - Export plugin descriptor
3. Build with `pts_add_plugin(NAME ... SOURCES ... DEPENDENCIES ...)`

**Host API for Plugins (`PtsHostApi`):**
Plugins receive a `PtsHostApi*` with access to:
- Logging: `create_logger()`, `log()`, `is_level_enabled()`
- Plugin queries: `get_plugin_handle()`, `query_interface()`
- Rendering APIs: `render_graph_api` (`PtsRenderGraphApi*`), `render_world_api` (`PtsRenderWorldApi*`)

**Renderer Plugin Interface (`RendererPluginInterfaceV1`):**
- `build_graph(host, frame, view, io)`: Called each frame to build render graph
- `on_resize(w, h)`: Handle output resolution changes
- `get_debug_outputs()`: Expose intermediate buffers for visualization
- `set_settings_blob()` / `get_settings_schema()`: Optional settings management

### USD Integration

Currently minimal - basic library linkage and tests only. USD is intended for:
- Scene graph serialization/deserialization
- Asset import/export pipeline
- Hierarchical scene representation

The scene system is undergoing refactoring (see comments in editor code).

### Platform Abstraction

**Windowing:**
- `IWindowing` interface with platform implementations (GLFW, null, Emscripten)
- `IViewport` provides native handles, DPI-aware extents, event signals
- Selected via `PTS_WINDOWING` CMake variable (set in `conanfile.py`)

**WASM/Emscripten:**
- Automatic detection via `EMSCRIPTEN` macro
- Plugins forced to static linking (`PTS_STATIC_PLUGINS=ON`)
- Boost configured as header-only to avoid compilation issues
- WebGPU backend: Browser-native (no Dawn dependency)
- GLFW backend: Emscripten's `-sUSE_GLFW=3` port
- Shell file: `web/index.html`

## Development Guidelines

### Invariant-Based Design
Many classes document their invariants (preconditions that must hold). Use `INVARIANT_MSG()` macro to enforce them in constructors. Examples:
- "m_webgpu_context is non-null and in Ready state"
- "Viewport has valid extents"

### Thread Safety
- `WebGpuContext` provides thread-safe handles to `Device` and `Surface`
- Plugin interface callbacks may be called from any thread (host synchronizes)
- `build_graph()` is always called on the render thread

### Error Handling
- Render graph API uses `PtsGraphError` codes (check return values)
- WebGPU errors logged via `ErrorScope` RAII wrapper
- Application initialization failures handled gracefully (show error window)

### Adding a New Renderer Plugin
1. Create plugin directory in `plugins/renderers/<name>/`
2. Implement `RendererPluginInterfaceV1`:
   - `build_graph()`: Import output texture, add passes, encode commands
   - `on_resize()`: Invalidate resolution-dependent resources
3. Use `PTS_PLUGIN_DEFINE()` to export descriptor
4. Add `add_subdirectory()` to `plugins/CMakeLists.txt`
5. Load in editor: `get_plugin_instance("your.plugin.id")`

### CMake Plugin Configuration
Use `pts_add_plugin()` function (from `cmake/PluginHelpers.cmake`):
```cmake
pts_add_plugin(
    NAME my_renderer_plugin
    SOURCES plugin.cpp renderer.cpp
    DEPENDENCIES optional_deps
)
```
This automatically:
- Creates shared library (or static if `PTS_STATIC_PLUGINS=ON`)
- Hides symbols by default (only `PTS_PLUGIN_EXPORT` visible)
- Links `core::api` for ABI-stable headers
- Sets output directory to `bin/plugins/`

### Code Style
- Format with clang-format: `.\pts.cmd format` (or `./pts format` on bash)
- C++17 standard
- Prefer RAII for resource management
- Use `pts::Signal<Signature>` for event propagation
- Namespace code under `pts::`

### Testing
- Unit tests in `core/tests/` using doctest framework
- Run with `.\pts.cmd test` (or `./pts test` on bash)
- Test plugins and render graph API independently

## Important Notes

### Dependencies
All dependencies managed via Conan (see `conanfile.py`):
- Core: fmt, spdlog, nlohmann_json, glm, stb, tinyobjloader, doctest, Boost
- Graphics: Dawn (WebGPU), GLFW (windowing), Slang (shader compilation)
- GUI: imgui (docking branch), imguizmo, imgui_color_text_edit, portable-file-dialogs
- Scene: OpenUSD

Custom Conan recipes in `tools/conan/` for packages not in Conan Center.

### Build Configurations
The build system uses per-configuration directories: `_build/Debug/`, `_build/Release/`, etc.
CMake is configured to NOT append configuration names to output paths (already organized by directory).

### Static vs Dynamic Plugins
- **Desktop**: Plugins built as shared libraries (`.dll`/`.so`/`.dylib`)
- **WASM**: Plugins statically linked into main executable (WASM doesn't support dynamic loading)
- Controlled by `PTS_STATIC_PLUGINS` CMake option (auto-enabled for Emscripten)

### Metaprogramming Patterns
The plugin system uses compile-time type lists and macros to generate ABI-safe wrappers:
- `PTS_METHOD()` generates functions that retrieve plugin instance from thread-local storage
- Enables polymorphic dispatch without virtual methods (safe across DLL boundaries)
- See `plugins/test_plugin/` for reference implementation
