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

### Embed Tool Resource Keys

The `embed` prebuild step generates C++ headers with `get_resource(key)` lookup. Resource keys are derived from input file paths by stripping the longest common prefix across all inputs in a group. Adding a new file to an embed group can change the common prefix and break existing lookups. When adding files to an embed resource group, always check that existing `get_resource()` callers still use the correct key.

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
3. Commit, push, and wait for CI to pass
4. Back in this project: `cd tools/framework && git checkout v<new-version>`
5. Commit the submodule pointer update

### Do not edit

These paths are generated or managed by the framework:

- `tools/framework/` — contribute upstream instead
- `tools/framework/_managed/` — generated venv, lockfile, pyproject
- `repo`, `repo.cmd`, `repo.ps1` — generated CLI shims
