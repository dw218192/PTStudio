# cpp-shader-compiler-tool

**Title:** Replace Python slangc.py with a C++ shader-compiler tool wrapping IShaderCompiler

**Status:** todo

**Prerequisite:** `linux-tool-builds` merged + closed. Adding another C++ build-time tool without Linux support would entrench the Windows-produces-artifacts CI antipattern we're trying to escape.

Consolidate shader compilation onto a single `IShaderCompiler` implementation used by both build-time prebuild and runtime. Kills Python/C++ duplication and enables dropping `slangc` from the native prebuild (unblocks `shader-variants-config` criteria #4 and #6 — descoped from that ticket).

## Context

Today:
- `tools/repo_tools/slangc.py` wraps libslang in Python, emits WGSL + reflect.json for both native and WASM prebuild.
- Runtime native uses `SlangCompiler` (C++, in `core/src/rendering/slangCompiler.cpp`) for on-demand compilation + disk cache.
- These are two independent libslang invocations with subtly different semantics — double maintenance surface.

Goal: one `IShaderCompiler` codepath, invokable as a CLI at build time.

## Scope

### New tool: `pts_shaderc`

- Lives under `tools/conan/pts_shaderc/` (new conan package, pattern after `usdz_pack`).
- Source depends on `core` (for `IShaderCompiler`, `SlangCompiler`) — or the shader-compiler code gets extracted into a small library that both `core` and `pts_shaderc` consume.
- CLI:
  ```
  pts_shaderc compile --source <path> --defines A,B --output <file.wgsl> [--reflect <file.reflect.json>]
  ```
- Emits WGSL + optionally reflect.json. Semantics identical to `SlangCompiler::compile()`.

### Prebuild replacement

- `tools/repo_tools/slangc.py` → deleted (or becomes a thin wrapper that just shells out to `pts_shaderc`).
- `config.yaml slangc:` section stays (schema unchanged); prebuild now invokes `pts_shaderc`.
- `shader_codegen` still consumes `*.reflect.json` — `pts_shaderc` emits these so that consumer is unchanged.

### Drop slangc from native prebuild

Once `pts_shaderc` is authoritative and reflect.json emission is covered, native prebuild no longer needs to emit WGSL — runtime `SlangCompiler` handles it. But it still needs reflect.json for `shader_codegen`.

Decide:
- **Option A (preferred)**: native prebuild runs `pts_shaderc --reflect-only` (no WGSL output) — minimal and aligns with the plan.
- Option B: fold reflect-json emission into `shader_codegen` directly (bigger refactor).

### `get_resource<WGSL>` direct callers

9 sites today call the embedded-resources API directly for WGSL bytes. On WASM this is fine (embed step still runs). On native, if we stop embedding WGSL, those callers break.

Options:
- **Route direct callers through `IShaderCompiler::compile()` (preferred)** so the compiler is the single source of shader text.
- Keep WGSL embedding on native (`pts_shaderc` emits, `embed` packs) but skip runtime use. Works but burns binary size.

## Acceptance criteria

- `pts_shaderc` conan package under `tools/conan/pts_shaderc/` builds on Windows and Linux
- CLI emits WGSL + reflect.json with byte-identical output to today's `slangc.py` (or documented differences)
- Python `slangc.py` deleted or reduced to a shim that invokes `pts_shaderc`
- Native prebuild no longer emits WGSL; reflect.json still produced (for `shader_codegen`)
- Emscripten prebuild still emits WGSL for embedding
- Direct `get_resource<WGSL>` callers routed through `IShaderCompiler` (or WGSL kept embedded with justification)
- Native Debug + Release build green without Python `slangc` prebuild step
- Emscripten Debug + Release build green
- `./repo test` green on native and WASM
- Hot-reload still works end-to-end
- Debug-MRT variant toggling (NO_DEBUG_TARGETS) still works

## Risks

- **Library vs executable**: `pts_shaderc` needs to link the shader-compiler code without dragging in all of `core`. May require extracting `IShaderCompiler` + `SlangCompiler` into a thin `core_shaderc` library.
- **Reflect.json schema drift**: Python `slangc.py` and C++ `SlangCompiler` may emit slightly different reflect.json today. Verify byte-compatibility before swapping — `shader_codegen` is sensitive to the schema.

## Out of scope

- New shader variant axes (PSO config, material features, etc.) — that's future work on top of `ShaderKey` (already landed in the squashed commit).
