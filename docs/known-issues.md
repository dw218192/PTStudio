# Known issues

Open gaps and limitations, with the evidence behind each one. Kept here rather
than in the README so the front page stays a showcase and this stays honest.

Last updated: 2026-08-15 (at the `develop` -> `main` merge).

---

## 1. The Forward renderer removes the D3D12 device on software adapters

**Status:** open, pre-existing, not caused by the rendering work merged in #33.

On a host with no real GPU (GitHub's `windows-2022` runners), the D3D12
software adapter drops the device a few seconds into the first Forward frame:

```
[I] [Editor] EditorApplication created
[I] [WebGPU] WebGPU device created successfully (Dawn backend)
[I] [Editor] Loaded scene: assets/scenes/test_cube.usda (1 objects)
[E] [WebGPU] [WebGPU Device Lost] Reason: Unknown,
    ID3D12Device::GetDeviceRemovedReason failed with DXGI_ERROR_DEVICE_REMOVED (0x887A0005)
```

### It is not a regression from the current branch

Measured by building and probing each commit, not inferred:

| Commit | Result |
|---|---|
| `develop` (control) | BAD - device removed, identical message and timing |
| `3e0aedc` area-light shadow maps | BAD |
| `d165efd` PCSS soft shadows | BAD |
| `a7e2215` fix shadow map bugs | BAD |
| `7c965d1` shadow-visibility pipeline | BAD |
| `HEAD` | BAD |

`develop` was green in April, so the trigger is environmental. The only things
that changed underneath a fixed `develop`:

- the `windows-2022` runner image moved to build 20260802
- Dawn is now compiled from source rather than served from a pre-April warm
  Conan cache

### Characterisation

- **Independent of scene content.** A 1-object `test_cube` (which has no lights
  at all) fails exactly like the 1788-prim Kitchen Set.
- **Independent of resolution.** 160x90 fails like 1280x720, so it is not
  per-pixel cost and not a TDR timeout.
- **Renderer-specific.** Wireframe and Path Trace both complete cleanly on the
  same adapter; only Forward dies. Diffing the frame graphs narrows the fault
  to the passes Forward adds.
- **Fails on the first submit**, after all textures, buffers and descriptors
  are created.

### Ruled out

- **No-debug MRT shader variant** - the "debug targets disabled" warning never
  fires on CI, so debug targets fit and that path never activates.
- **Unbounded shader loops** - every loop in the shadow/SSAO/contact-shadow
  shaders is bounded by a compile-time constant or a uniform.
- **Uniform buffer size mismatch** - the gbuffer binding was correctly updated
  from 128 to 192 bytes within its 256-byte alignment.

### Workaround in place

`editorSmoke_*` is gated behind `PTSTUDIO_SKIP_EDITOR_SMOKE`, set for
`windows-x64` in CI (see `tools/repo_tools/launch.py` and
`.github/workflows/ci.yml`). `ptSmoke_*` (path tracer) still runs there and
still covers device, shader, BVH and readback end to end. Local runs are
unaffected: unset gives 50 tests, set gives 41.

### Next step

Enable the D3D12 debug layer and DRED breadcrumbs through Dawn instance
toggles (`WGPUDawnTogglesDescriptor`) to name the exact faulting operation.
Note this cannot be reproduced locally on a machine with a real GPU: Dawn
returns `Unavailable` for `forceFallbackAdapter`, so iteration means ~1.5h CI
cycles.

---

## 2. image-diff exceeds thresholds against path-traced ground truth

**Status:** open.

Measured locally against the committed GT:

| Case | Score (max tile-mean FLIP) | Threshold |
|---|---|---|
| `area_light_pcss` | 0.60373 | 0.30000 |
| `brdf_ibl` | 0.75854 | 0.30000 |

Roughly 2x over. Thresholds were **not** raised and GT was **not** rebaked:
per project convention a failing image-diff after raster changes is a raster
quality gap to fix at the source, not something to mask by moving the goalposts.

Plausibly shares a cause with the shadow work, since the divergence arrived
with the same commits. Investigate with `./repo image-diff` and the FLIP
heatmaps in `_test_captures/`.

---

## 3. The image-diff CI job is permanently red

**Status:** open, consequence of issue 1.

`image-diff` launches the editor with the Forward renderer, so on CI it fails
for the adapter reason above rather than for any perceptual difference. The job
is `continue-on-error: true` so it does not block a merge, but a permanently
red job trains people to ignore it, and its failure message is misleading.

Options: gate it the same way `editorSmoke_*` is gated, or move it to a host
with a real GPU.

---

## 4. Windows CI is pinned to `windows-2022`

**Status:** open, temporary shelter.

`windows-latest` now resolves to the `windows-2025-vs2026` image (Visual Studio
2026 / VS 18). Dawn 20251002's vendored SPIRV-Tools cannot compile under its
MSVC 14.51: SPIRV-Tools builds with `/WX`, and 14.51 raises a new C5232 ("in
C++20 this comparison calls operator== recursively") in `util/small_vector.h`
that older MSVC did not.

The Windows jobs are therefore pinned to `windows-2022`, which also matches the
VS 2022 developers build with locally, so CI and local machines resolve to the
same Conan package IDs.

GitHub will eventually retire that image. The durable fix is upgrading Dawn to
a version whose vendored SPIRV-Tools is clean under 14.51 -- but per the
version-pinning invariant in `CLAUDE.md`, the Dawn version pins the
emdawnwebgpu port version, so that upgrade moves the WASM build in lockstep.

---

## 5. Packaged Conan run-environment scripts bake in absolute build-machine paths

**Status:** open, latent portability bug.

`_package/<platform>/Release/bin/conanrunenv-*.bat` contains absolute paths
into the Conan cache of whichever machine produced the package, e.g.
`D:\a\PTStudio\PTStudio\_build\windows-x64\deps\full_deploy\...`. A CI probe
confirmed those directories do not exist on a different runner:

```
MISSING: D:\a\...\full_deploy\host\dawn\20251002.162335\Release\x86_64\bin
MISSING: D:\a\...\full_deploy\host\openusd\25.11-dev\Release\x86_64\lib
```

`launch.py` prefers that env script over the self-contained `_package/*/deps/`
directory (see the `_resolve_env_script` branch), so a package is not portable
across machines by that path. CI happens to work because the build and test
jobs land on equivalent layouts, and because `deps/` is used as a fallback.

Related: `CLAUDE.md` already documents the `full_deploy` invariant this stems
from.

---

## 6. Clustered lighting is not implemented

**Status:** open, by design for now.

The forward light iteration path loops over every light in the scene
(`core/shaders/light_iteration_lib.slang`, "V1: iterates all lights"). Shadow
allocation for many lights is deferred to clustered lighting rather than being
solved with ad-hoc priority sorting. `assets/scenes/clustered_lighting_test.usda`
exists as a test scene for the eventual implementation, not as evidence that
clustering exists.

---

## 7. The frame graph does not alias resources or automate barriers

**Status:** open, scope limitation.

Passes declare textures, buffers and descriptor sets up front, and the graph
compiles those declarations into concrete GPU resources with strong-typedef
integer handles and implicit liveness. It does **not** implement transient
memory aliasing or barrier placement -- those are left to the backend. Worth
stating explicitly because "frame graph" often implies both.

---

## 8. `--camera` does not resolve prims authored in a `--usd-override` layer

**Status:** open, observed but not root-caused.

Authoring a new `Camera` prim in an override layer and selecting it with
`--camera /Root/ShowcaseCam` silently falls back to the default view. The
override itself is applied (the editor logs "Applied USD override"), and the
same flow works for cameras authored in the base stage.

Consequence: the glass/transmission scene has no gallery image, because it has
no camera in its base stage and the override camera would not take.

Workaround: author the camera in the scene file itself.

---

## 9. CodeRabbit skips review on large PRs

**Status:** informational.

CodeRabbit declines PRs above 100 changed files ("Review skipped: 105 files
exceed the limit of 100"). PR #33 therefore merged without bot review. Worth
knowing when landing large branches -- split them if bot review matters.

---

## 10. Emscripten Debug builds are impractical

**Status:** accepted limitation.

Debug WASM binaries exceed 1 GB. Only `--build-type Release` is supported for
the Emscripten platform.
