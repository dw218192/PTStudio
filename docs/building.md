# Building

## Quick start

```bash
bash tools/framework/bootstrap.sh   # one-time: sets up the hermetic tool environment
./repo build                        # native
./repo test
```

Web build:

```bash
./repo build --platform emscripten --build-type Release
```

Only Release is supported for Emscripten -- Debug WASM binaries exceed 1 GB and
are impractical.

Run `./repo --help` for the full command set (build, test, format, package,
publish, image-diff, launch, and the shader/asset prebuild tools).

## Prerequisites

- A C++17 toolchain (MSVC, Clang, or GCC)
- A GPU driver with Vulkan or D3D12 support (Windows and Linux are the tested
  native targets)
- Python 3.12+ (the bootstrap script handles the rest)

## Dependencies

Dependencies are managed with Conan. Packages not on Conan Center are built
from local recipes in `tools/conan/`, which are auto-discovered and exported
before each build -- changing a recipe invalidates the Conan cache for that
package. Lock files (`conan_glfw.lock` for native, `conan_emscripten.lock` for
wasm) are committed for reproducible builds; regenerate with `./repo build -u`.

On Windows with a cold `CONAN_HOME`, the default profile is written pinned to
MSVC rather than detected. `conan profile detect` is not deterministic there:
with a GCC toolchain on PATH and a Visual Studio it does not recognise, it
resolves `compiler=gcc`, and Dawn then fails on MSVC-only CRT macros. The
Visual Studio version is looked up through `vswhere` at runtime so this keeps
working across runner and toolchain migrations. See
`tools/repo_tools/build/conan.py`.

## Tooling

The project uses [repokit](../tools/framework/README.md) for build, test and
format tooling, driven by `config.yaml` at the repo root. Build-time tools come
in two flavours:

- **Python tools** in `tools/repo_tools/`, invoked as `./repo <tool>` -- they
  run in the managed venv and need no compilation.
- **C++ tools** in `tools/conan/<tool>/` as standalone Conan packages (e.g.
  `usdz_pack`). These cannot cross-compile to WASM, so Emscripten builds
  consume the outputs they produce rather than invoking them directly.

For platform-specific build gotchas (OpenUSD + TBB on Emscripten, the Conan
`full_deploy` invariant, Tracy's shutdown deadlock), see `CLAUDE.md`.

## CI

The Windows jobs are pinned to `windows-2022` rather than `windows-latest`.
See [known-issues.md](known-issues.md#4-windows-ci-is-pinned-to-windows-2022).
