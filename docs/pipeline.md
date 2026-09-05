# Shader and asset pipeline, and how correctness is checked

## Shaders

Shaders are authored once in **Slang** and compiled to WGSL at build time, with
reflection JSON emitted alongside for descriptor generation. Bind group layouts
are generated from that reflection data rather than hand-written and kept in
sync by hope.

Build-time preprocessor variants are declared in `config.yaml` and code
generated into a C++ registry. The main example is a `_no_debug` variant of
each scene shader, which drops the debug MRT outputs on devices whose
`maxColorAttachmentBytesPerSample` budget cannot fit them -- the WebGPU spec
prices RGBA8Unorm at 8 bytes per sample, not 4, so five attachments cost 40
bytes and exceed the 32-byte limit reported by instrumented runtimes such as
RenderDoc and NSight.

Passes query device limits during `setup()` and compute an all-or-nothing
allowed debug-target count; `load_pass_shader_module()` then selects the
no-debug module automatically. On native, uncached variants are recompiled
through libslang at runtime and cached to `<exe-dir>/shader_cache/`; on the web
the precompiled variants are embedded in the binary.

Shader authoring conventions (Slang `mul` and matrix constructor semantics,
visibility modifiers, the `#ifndef NO_DEBUG_TARGETS` guard) are documented in
`CLAUDE.md`.

## Assets

Scenes are **OpenUSD** stages. The full USD runtime -- including plugin
discovery, schema registration, and TBB -- is static-linked into the WASM
build, which required working through constructor dead-stripping and static
initialisation ordering problems; those findings are written up in `CLAUDE.md`.

Scenes are packaged to USDZ at build time by a native host tool (`usdz_pack`)
and embedded into the web build.

## Verification

Rendering regressions are silent and easy to ship, so correctness is checked
mechanically rather than by eye.

### Golden-image regression suite

Reference images are baked with the path tracer at 4096 spp, and the forward
renderer is diffed against them using **FLIP**, a perceptual metric. The score
is the *maximum over per-tile mean* FLIP rather than the whole-image mean,
because a whole-image average lets a badly wrong shadow in a mostly-dark scene
hide in the error-free background. Cases, thresholds and the GT bake settings
live under `image_diff:` in `config.yaml`.

```bash
pixi run image-diff
```

A failing case after a raster change is treated as a raster/GT quality gap to
fix at the source. Thresholds are not raised and GT is not rebaked to make a
divergence disappear. Current failures are recorded in
[known-issues.md](known-issues.md#2-image-diff-exceeds-thresholds-against-path-traced-ground-truth).

### Headless capture

```
pixi run launch editor --capture-and-quit[=output.png] [--usd scene.usda] [--frames 5] \
                     [--renderer Forward] [--debug-output "Direct Diffuse"] \
                     [--camera /Root/Camera] [--usd-override override.usda]
```

Renders a scene to PNG with no window, optionally selecting a camera, renderer,
or named debug target. Captures default to `_captures/<timestamp>.png`; output
is 1280x720 RGBA8 and excludes editor-only passes (grid, gizmo, overlay).
Every rendering change gets verified against a picture, not against a green
build.

When a camera or scene has changed, capture a temporary matching path-traced
reference and compare it without replacing the committed golden:

```bash
pixi run launch editor --usd assets/scenes/area_light_test.usda --camera /Root/Camera_3 --renderer "Path Trace" --frames 1024 --capture-and-quit=_test_captures/area_camera3_pt.png
pixi run image-diff --case area_light_pcss --reference _test_captures/area_camera3_pt.png
```

Add `--capture path/to/existing.png` to compare an existing screenshot without
launching again. Both options require `--case`. The JSON summary records the
reference used; the committed thresholds still apply.

### Area-light shadow approximation

Rect and disk shadows use a depth cube at the emitter center. Every PCSS tap
selects its own cube face, retaining coverage when a light moves sideways.
Area-only scenes use six 1024-square faces (24 MiB); arrays shared with distant
lights retain their 2048-square resolution. The current forward lighting path
still resolves visibility for one light.

The filter uses 32 blocker taps and 64 visibility taps, with a stratified
emitter-shaped kernel and temporal sample shifts. Depth comparisons intersect
the receiver plane at the actual texel center. Geometric normals come from
nearby depth samples chosen to avoid crossing silhouettes. Samples are weighted
by the emitter's diffuse geometric contribution, including the receiver horizon.

This remains a single-point PCSS approximation: hidden blockers and surfaces
not seen from the emitter center cannot be recovered, specular visibility shares
the diffuse estimate, and broad kernels can leak light. The scene's artistic
`pts:shadow:pcss:softness` control scales the emitter axes; `area_light_test` now
uses 0.65 instead of the old disk filter's 2.0. Camera 3's edited transform is
preserved. [PCSS reference](https://developer.download.nvidia.com/assets/gamedev/docs/PCSS_Integration.pdf)
and [cube shadow-map reference](https://developer.nvidia.com/gpugems/gpugems/part-ii-lighting-and-shadows/chapter-12-omnidirectional-shadow-mapping).

Native demo selection opens USDA sources when available. Saving a plain layer
rebases its asset paths without extracting duplicates; imports from USDZ keep
references into the original package, which must remain available. USDZ export
continues to produce a packaged asset.

### CI

Format check, native and Emscripten builds, and tests on both, on every push.
The image diff runs alongside as a non-blocking signal, so a perceptual
regression is visible without a threshold wobble blocking a merge.

The `editorSmoke_*` tests are skipped on Windows CI because the runners'
software adapter cannot run the Forward pipeline; `ptSmoke_*` still covers the
GPU path there. See
[known-issues.md](known-issues.md#1-the-forward-renderer-removes-the-d3d12-device-on-software-adapters).
