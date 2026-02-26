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

## Code Conventions

- C++17, `webgpu.h` API for rendering (same header for Dawn and emdawnwebgpu)
- On Emscripten, use `IMGUI_IMPL_WEBGPU_BACKEND_DAWN` (emdawnwebgpu IS Dawn)
- Dawn-only APIs (e.g. `wgpuDeviceGetAdapter`) must be guarded with `#ifndef __EMSCRIPTEN__`
- emdawnwebgpu async APIs are JS Promises; synchronous busy-wait loops deadlock on Emscripten
