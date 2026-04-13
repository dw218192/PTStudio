# PTStudio

C++17 scene editor with WebGPU rendering. Builds natively (Windows/Linux via Dawn) and for the web (Emscripten via emdawnwebgpu).

## Build System

Uses the [repokit](tools/framework/README.md) framework. See that README for CLI usage, `config.yaml` schema, tokens, and dimensions.

**Local Conan recipes** in `tools/conan/` are auto-discovered and exported before each build. Changing a recipe invalidates the Conan cache for that package.

**Lock files**: `conan_glfw.lock` (native), `conan_emscripten.lock` (wasm). Regenerate with `./repo build -u`.

### Conan `full_deploy` Invariant

`conan install --deployer=full_deploy` copies packages into `_build/<platform>/deps/` and rewrites `buildenv_info` paths to the deploy folder. But `conf_info` (CMake toolchain, compiler paths) stays pointing to the original Conan package. On Windows CI (workspace on D:, Conan cache on C:) these end up on different drives. Never export env vars from Conan recipes that must resolve to the same root as the compiler — let the tool derive them from its own install location instead.

### Emscripten Build

- emsdk is a Conan `tool_requires`, Dawn version pins the emdawnwebgpu port version
- The emsdk recipe does NOT export `EM_CACHE` or `EM_CONFIG` to consumers — emscripten defaults to `<EMSCRIPTEN_ROOT>/cache/`, which is always on the same drive as `em++`

### OpenUSD + TBB on Emscripten

Static-linking OpenUSD via Conan on Emscripten has several non-obvious failure modes:

- **Constructor dead-stripping**: OpenUSD's plugin discovery relies on `Plug_InitConfig`, an `__attribute__((constructor))` in `initConfig.cpp`. When USD libraries are separate static `.a` archives (Conan components), the linker drops `initConfig.o` because nothing references its symbols. Fix: `--whole-archive` on `libusd_plug.a` (see `CMakeLists.txt`).
- **TBB static init crashes**: Setting `PXR_WORK_THREAD_LIMIT` to non-zero forces `tbb::global_control` creation during `__wasm_call_ctors`, before TBB's function table is ready. Leave it at default (0).
- **TBB + EMSCRIPTEN_WITHOUT_PTHREAD**: The Conan profile passes `-pthread` globally, so TBB source sees `__EMSCRIPTEN_PTHREADS__`. Don't override with `EMSCRIPTEN_WITHOUT_PTHREAD` — it creates contradictory state.
- **"Cannot create a log file"**: A misleading secondary error from USD's crash handler. The real error is whatever triggered the abort; this message means `ArchGetTmpDir()` failed to create a temp file on the WASM virtual filesystem.
- **Plugin resources**: Embed full `resources/` directories (not just `plugInfo.json`) — `generatedSchema.usda` is required for type registration.

### Prebuild Tool Config

Tool configs (slangc, shader_codegen, embed) live at the top level of `config.yaml`. The `build.prebuild` section just lists the tools to run (as empty dicts `{}`). When invoked — whether standalone or as a prebuild step — `invoke_tool` reads the top-level config via `config.get(tool_name, {})`.

### Embed Tool Resource Keys

The `embed` prebuild step generates C++ headers with `get_resource(key)` lookup. Resource keys are derived from input file paths by stripping the longest common prefix across all inputs in a group. Adding a new file to an embed group can change the common prefix and break existing lookups. When adding files to an embed resource group, always check that existing `get_resource()` callers still use the correct key.

### Tracy Profiler (debug builds only)

Tracy 0.13.1's static `s_profiler` deadlocks at process exit on Windows if `<thread>` is included in widely-used headers — the changed static init ordering causes Tracy's destructor to run after WinSock cleanup, and its profiler thread hangs in `accept()`. **Never include `<thread>` (or headers that transitively include it, like `backgroundTask.h`) in `.h` files that are widely included.** Forward-declare and include in `.cpp` only. The proper fix is rebuilding Tracy with `TRACY_DELAYED_INIT=ON` + `TRACY_MANUAL_LIFETIME=ON`.

## Visual Verification

Use `--capture-and-quit` to verify rendering changes without manual inspection:

```
./repo launch editor --capture-and-quit[=output.png] [--usd scene.usda] [--frames 5] \
                     [--renderer Forward] [--debug-output "Direct Diffuse"] \
                     [--usd-override override.usda]
```

- Captures default to `_captures/<timestamp>.png` when no path is given
- `--frames N` lets async loads settle before capture (default: 1)
- `--debug-output` captures a named debug target instead of scene_color
- Editor passes (grid, gizmo, overlay) are excluded from capture output
- Output is always 1280x720 RGBA8

## Verification

Never declare a feature "working" based on build/test passing alone. For runtime behavior (rendering, hot-reload, UI), always launch the application (`./repo launch editor`) and verify visually or via log output before concluding and committing. Add diagnostic logging when needed to confirm correctness — guessing at root causes from code alone leads to wasted cycles. `./repo launch editor` returns the editor's log output directly — use it.

## Debug MRT Targets & Device Limits

Scene passes can declare debug MRT outputs (Normals, Base Color, etc.) via `debug_target_names()`. These are gated at runtime by `maxColorAttachmentBytesPerSample` — the WebGPU spec's `renderTargetPixelByteCost` for RGBA8Unorm is 8 bytes (not 4), so 5 attachments cost 40 bytes, exceeding the 32-byte limit on instrumented runtimes (RenderDoc, NSight).

**How it works:**
- `IScenePass::setup()` queries device limits and computes an all-or-nothing `m_allowed_debug_count` (all debug targets fit, or none)
- `effective_debug_target_names()` returns the gated count; the editor UI and frame graph use this
- `load_pass_shader(resource_key)` automatically selects the no-debug shader variant when targets are disabled — passes just call this instead of `ShaderLoader::load()` directly
- The no-debug variant is compiled at build time with `-DNO_DEBUG_TARGETS` (see `config.yaml` slangc entries with `defines:`)
- On native, `SlangCompiler` recompiles via libslang with the define and caches the WGSL on disk (`<exe-dir>/shader_cache/`); on WASM the `EmbeddedCompiler` serves the pre-compiled embedded variant.

**Shader convention:** guard debug MRT struct fields and writes with `#ifndef NO_DEBUG_TARGETS`. The variant key is derived automatically by inserting `_no_debug` before the extension (e.g. `forward.wgsl` → `forward_no_debug.wgsl`). Both the base and variant WGSL must be listed in `config.yaml` under `slangc.shaders` and `embed.resources`.

## Slang Shader Conventions

### GLSL→Slang porting: `mul` and matrix constructors

Slang `float3x3(A, B, C)` passes A, B, C directly to WGSL `mat3x3(A, B, C)`, which interprets them as **columns** (not rows). When porting GLSL code that constructs a matrix with `mat3(col0, col1, col2)`, use the same arguments in Slang — they'll arrive as columns in WGSL unchanged.

For matrix-vector multiplication: `mul(M, v)` = `M * v`, `mul(v, M)` = `v * M`. When porting GLSL `M * v` where M was built with column arguments, use `mul(v, M)` in Slang — the column-as-column constructor plus row-vector multiply gives the correct result.

### Visibility modifiers

Default visibility is `public`, but once ANY declaration uses an explicit modifier (`internal`, `public`, `private`), all non-annotated declarations become `internal`. To use `internal` on helpers, explicitly mark the public API surface with `public` — including struct fields.

## Code Conventions

- C++17, `webgpu.h` API for rendering (same header for Dawn and emdawnwebgpu)
- On Emscripten, use `IMGUI_IMPL_WEBGPU_BACKEND_DAWN` (emdawnwebgpu IS Dawn)
- Dawn-only APIs (e.g. `wgpuDeviceGetAdapter`) must be guarded with `#ifndef __EMSCRIPTEN__`
- emdawnwebgpu async APIs are JS Promises; synchronous busy-wait loops deadlock on Emscripten

## Repo tooling

This project uses [repokit](tools/framework/README.md) for general project tooling (e.g. build, test, format).

- **CLI**: `./repo <command>` (or `repo.cmd` on Windows). Run `./repo --help` to discover commands.
- **Config**: `config.yaml` at the project root.
- **Framework path**: `tools/framework/`

### Contributing to the framework
1. `cd tools/framework && git fetch origin && git switch main && git pull --ff-only origin main`
2. Make changes, bump the version in `pyproject.toml`, add a `CHANGELOG.md` entry
3. Commit, push, and wait for CI to pass (CI auto-tags `v<version>` from `pyproject.toml`)
4. Back in this project: `cd tools/framework && git checkout v<new-version>`
5. Commit the submodule pointer update

### Do not edit

These paths are generated or managed by the framework:

- `tools/framework/` — contribute upstream instead
- `tools/framework/_managed/` — generated venv, lockfile, pyproject
- `repo`, `repo.cmd`, `repo.ps1` — generated CLI shims
