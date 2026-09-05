# Building

## Quick start

Install [pixi](https://pixi.sh), then:

```bash
pixi run build      # native
pixi run test
```

pixi brings its own Python and the entire tool environment (Conan, CMake,
Ninja, clang-format, ruff), so there is nothing to bootstrap first and no
system Python to match.

Web build:

```bash
pixi run build --platform emscripten --build-type Release
```

Only Release is supported for Emscripten -- Debug WASM binaries exceed 1 GB and
are impractical.

## Tasks

| Task | What it does |
|---|---|
| `pixi run build` | Build the project (prebuild chain + CMake) |
| `pixi run test` | Run the test suite |
| `pixi run fmt` | Format sources in place |
| `pixi run lint` | Check formatting without modifying files |
| `pixi run check` | `lint` + `test` |
| `pixi run package` | Collect build outputs into `_package/` |
| `pixi run publish` | Prepare the static site from packaged WASM artifacts |
| `pixi run image-diff` | Diff renderer captures against golden GT via FLIP |
| `pixi run launch` | Launch a built executable |

Tasks forward extra arguments, e.g.
`pixi run build --platform emscripten --build-type Release`.

Asset-generation and maintenance tasks are also direct commands:
`pixi run slangc`, `pixi run shader-variants-codegen`, `pixi run embed`,
`pixi run usdz`, `pixi run bake-gt`, and `pixi run clean`.
Use `pixi task list` to list tasks and `pixi run <task> --help` for options.
`pixi run test-tasks` checks the Python task implementations.

## Prerequisites

- A C++17 toolchain (MSVC, Clang, or GCC) -- this is the one thing pixi does
  not provide
- A GPU driver with Vulkan or D3D12 support (Windows and Linux are the tested
  native targets)

## Dependencies

Dependencies are managed with Conan. Packages not on Conan Center are built
from local recipes in `tools/conan/`, which are auto-discovered and exported
before each build -- changing a recipe invalidates the Conan cache for that
package. Lock files (`conan_glfw.lock` for native, `conan_emscripten.lock` for
wasm) are committed for reproducible builds; regenerate with `pixi run build -u`.

On Windows with a cold `CONAN_HOME`, the default profile is written pinned to
MSVC rather than detected. `conan profile detect` is not deterministic there:
with a GCC toolchain on PATH and a Visual Studio it does not recognise, it
resolves `compiler=gcc`, and Dawn then fails on MSVC-only CRT macros. The
Visual Studio version is looked up through `vswhere` at runtime so this keeps
working across runner and toolchain migrations. See
`tasks/build_support/conan.py`.

## Task implementation

Each task in `pixi.toml` directly runs a module under `tasks/`, for example
`python -m tasks.build`. There is no umbrella command or dynamic command
registry. `tasks/utils/` contains shared configuration and subprocess helpers;
`tasks/build_support/` contains Conan and CMake support.

`config.yaml` holds asset lists, package mappings, and build settings. The
`paths` mapping defines output paths; `{platform}` and `{build_type}` expand
from the command's options. Optional `config.local.yaml` values override the
project settings. Platform-specific keys such as `mappings@emscripten` retain
their existing behavior.

The build calls formatting, shader compilation, variant generation, USDZ
packaging, and embedding directly, in that order. Native host binaries are
built before the asset steps. `--build-only` skips configuration and asset
generation; `--host-tools-only` builds native helpers for the web build.

After changing the Python environment, run the full `pixi run build` for each
platform/configuration once to regenerate CMake's cached executable paths.
Existing `--build-only` directories can still point at a removed environment.

- **Python tasks** run inside Pixi and need no compilation.
- **C++ tools** in `tools/conan/<tool>/` are standalone Conan packages (for
  example `usdz_pack`). Emscripten consumes their generated outputs.

For platform-specific build gotchas (OpenUSD + TBB on Emscripten, the Conan
`full_deploy` invariant, Tracy's shutdown deadlock), see `CLAUDE.md`.

## CI

The Windows jobs are pinned to `windows-2022` rather than `windows-latest`.
See [known-issues.md](known-issues.md#4-windows-ci-is-pinned-to-windows-2022).
