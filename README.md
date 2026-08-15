# PTStudio

A real-time and offline renderer built on **WebGPU** and **OpenUSD**, written in C++17. One codebase, one shader source, two targets: a native desktop app (Dawn) and a browser build (Emscripten + emdawnwebgpu).

It started as a hobby project to learn modern graphics, and turned into a playground for implementing rendering techniques end to end -- from the shader math up through the frame graph, the asset pipeline, and the regression harness that keeps them honest.

**[Try it in your browser](https://dw218192.github.io/PTStudio/editor/)** -- no install, runs on WebGPU.

> Requires a WebGPU-capable browser -- a recent Chrome or Edge is the safest bet.

---

## Gallery

![Path-traced camera scene](docs/readme_assets/pathtraced_camera.png)

*Progressive path tracer. Physically-based materials, IBL from an HDR environment, and a BVH-accelerated compute traversal.*

![Pixar Kitchen Set](docs/readme_assets/kitchen_set.png)

*Pixar's Kitchen Set (1788 prims) loaded straight from USD and rendered in real time -- geometry, texture binding, and material resolution all driven by the USD stage.*

<table>
<tr>
<td width="50%"><img src="docs/readme_assets/brdf_ibl.png" alt="BRDF and IBL sweep"></td>
<td width="50%"><img src="docs/readme_assets/ltc_area_light.png" alt="LTC area light with PCSS shadows"></td>
</tr>
<tr>
<td><em>Roughness/metallic sweep under image-based lighting -- split-sum specular with a prefiltered environment and BRDF LUT.</em></td>
<td><em>LTC area light with PCSS soft shadows. Penumbra width tracks the emitter's size, and the visibility signal is temporally resolved with variance clamping.</em></td>
</tr>
<tr>
<td colspan="2"><img src="docs/readme_assets/many_lights.png" alt="Many-light scene"></td>
</tr>
<tr>
<td colspan="2"><em>Many-light scene exercising the forward light iteration path, which currently loops over every light in the scene. Clustered light assignment is the next step here.</em></td>
</tr>
</table>

---

## Two renderers, one scene representation

PTStudio ships two interchangeable renderers over a shared USD-backed scene. You can flip between them at runtime on the same stage, which makes the rasterizer directly comparable against a ground-truth reference.

| | |
|---|---|
| **Forward** | Forward renderer with a G-buffer prepass, analytic area lights, shadow maps, screen-space effects, and IBL. This is the real-time path. |
| **Path Trace** | Progressive BVH path tracer running as a compute pass. Same materials, same lights, same stage -- used both as a visual target and as the source of truth for the automated image-diff suite. |

## Techniques implemented

### Lighting and materials

- **LTC area lights** -- linearly transformed cosines for rect and disk lights, giving analytic, noise-free specular and diffuse response from area sources rather than punctual approximations.
- **Cook-Torrance PBR** with a GGX/Smith BRDF, metallic-roughness workflow, normal mapping, and USD `UsdPreviewSurface` material binding.
- **Image-based lighting** -- equirectangular-to-cubemap projection, irradiance convolution for the diffuse term, roughness-prefiltered specular mips, and a precomputed split-sum BRDF LUT.
- **Glass and transmission** -- refractive materials shared between the raster and path-traced paths.

### Shadows

This is where most of the work went.

- **PCSS soft shadows** -- percentage-closer soft shadows with blocker search, for distant (directional) lights as well as rect and disk area lights, so penumbra width tracks the actual emitter size.
- **Receiver-plane depth bias** -- derivative-based bias propagated through the light's clip space, which is what makes PCSS survive grazing angles without acne or peter-panning.
- **Split visibility pipeline** -- shadow visibility is generated and resolved in separate passes rather than sampled inline, decoupling shadow cost from shading and making the visibility signal available to temporal filtering.
- **Temporal resolve with variance clamping** -- history is reprojected using G-buffer motion vectors and clamped against the local neighborhood's variance, which suppresses the noise of a sparse PCSS kernel without smearing on motion.
- **Contact shadows** -- screen-space ray marching to recover the short-range occlusion that shadow-map resolution drops.

### Screen space and post

- **SSAO** with a bilateral (depth-aware) blur.
- **G-buffer motion vectors** driving history reprojection.
- **Tone mapping** as the final graph node.

### Architecture

- **Frame graph** -- passes declare their textures, buffers, and descriptor sets up front through a builder, and the graph compiles those declarations into concrete GPU resources. Everything is referenced by strong-typedef integer handles with implicit liveness; names exist only as debug labels.
- **Shader-driven descriptors** -- bind group layouts are generated from Slang reflection data rather than hand-written and kept in sync by hope.
- **Async scene preparation** -- USD stage traversal and CPU-side data prep run off the render thread, feeding an immutable `PreparedSceneData` snapshot to the renderer.

## The shader and asset pipeline

Shaders are authored once in **Slang** and compiled to WGSL at build time, with reflection JSON emitted alongside for descriptor generation. Build-time preprocessor variants are declared in `config.yaml` and code-generated into a C++ registry -- for example, a `_no_debug` variant of each scene shader that drops the debug MRT outputs when the device's `maxColorAttachmentBytesPerSample` budget can't fit them (which happens under RenderDoc and NSight). On native, uncached variants are recompiled through libslang at runtime; on the web, the precompiled variants are embedded in the binary.

Scenes are **OpenUSD** stages. The full USD runtime -- including plugin discovery, schema registration, and TBB -- is static-linked into the WASM build, which required solving a genuinely unpleasant set of problems around constructor dead-stripping and static-init ordering (written up in `CLAUDE.md`). Scenes are packaged to USDZ at build time by a native host tool and embedded into the web build.

## Verifying that it actually works

Rendering regressions are silent and easy to ship, so correctness is enforced mechanically:

- **Golden-image regression suite** -- reference images are baked with the path tracer at 4096 spp, and the forward renderer is diffed against them using **FLIP**, a perceptual metric. The score is the *maximum over per-tile mean* FLIP rather than the whole-image mean, because a whole-image average lets a badly wrong shadow in a mostly-dark scene hide in the error-free background.
- **Headless capture** -- `./repo launch editor --capture-and-quit` renders a scene to PNG with no window, optionally selecting a camera, renderer, or named debug target. Every rendering change gets verified against a picture, not against a green build.
- **Full CI matrix** -- format check, native and Emscripten builds, and tests on both, run on every push. The image diff runs alongside them as a non-blocking signal, so a perceptual regression is visible without a threshold wobble being able to block a merge.

## Build

```bash
bash tools/framework/bootstrap.sh   # one-time: sets up the hermetic tool environment
./repo build                        # native
./repo test
```

Web build:

```bash
./repo build --platform emscripten --build-type Release
```

Only Release is supported for Emscripten -- Debug WASM binaries exceed 1 GB and are impractical.

Run `./repo --help` for the full command set (build, test, format, package, publish, image-diff, launch, and the shader/asset prebuild tools).

### Prerequisites

- A C++17 toolchain (MSVC, Clang, or GCC)
- A GPU driver with Vulkan or D3D12 support (Windows and Linux are the tested native targets)
- Python 3.12+ (the bootstrap script handles the rest)

Dependencies are managed with Conan; packages not on Conan Center are built from local recipes in `tools/conan/`. Lock files are committed for reproducible builds.
