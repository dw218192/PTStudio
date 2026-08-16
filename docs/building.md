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

`pixi run repo --help` reaches the full command set, including tools without a
task shortcut (`slangc`, `embed`, `usdz`, `clean`, `context`, ...).

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
wasm) are committed for reproducible builds; regenerate with `./repo build -u`.

On Windows with a cold `CONAN_HOME`, the default profile is written pinned to
MSVC rather than detected. `conan profile detect` is not deterministic there:
with a GCC toolchain on PATH and a Visual Studio it does not recognise, it
resolves `compiler=gcc`, and Dawn then fails on MSVC-only CRT macros. The
Visual Studio version is looked up through `vswhere` at runtime so this keeps
working across runner and toolchain migrations. See
`tools/repo_tools/build/conan.py`.

## Tooling

pixi owns the environment and the task entry points (`pixi.toml` +
`tasks/*.py`). It replaced repokit's bootstrap scripts, its generated venv
under `tools/framework/_managed/`, and the generated `./repo` shim -- repokit
itself is deprecated upstream, so only the driver moved; the tool
implementations are unchanged and still read `config.yaml` at the repo root.

- `tasks/repo.py` puts both tool trees on `sys.path` and hands over to the
  existing CLI. It is the pixi-era replacement for the `./repo` shim.
- `tools/repo_tools/` holds the project-owned tools.
- `tools/framework/repo_tools/` is repokit, now imported as a plain library
  rather than bootstrapped.

Build-time tools come in two flavours:

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
