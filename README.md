# PTStudio

A real-time and offline renderer built on **WebGPU** and **OpenUSD**, written in C++17. One codebase, one shader source, two targets: a native desktop app (Dawn) and a browser build (Emscripten + emdawnwebgpu).

Mainly a personal hobby project and rendering playground.

**[Try it in your browser](https://dw218192.github.io/PTStudio/editor/)**

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

## Renderers

PTStudio ships two interchangeable renderers over a shared USD-backed scene. You can flip between them at runtime on the same stage, which makes the rasterizer directly comparable against a ground-truth reference.

| | |
|---|---|
| **Forward** | Forward renderer with a G-buffer prepass, analytic area lights, shadow maps, screen-space effects, and IBL. This is the real-time path. |
| **Path Trace** | Progressive BVH path tracer running as a compute pass. Same materials, same lights, same stage -- used both as a visual target and as the source of truth for the automated image-diff suite. |

## Real-time Rendering Techniques implemented

### Lighting and materials

- **LTC area lights** -- linearly transformed cosines for rect and disk lights, giving analytic, noise-free specular and diffuse response from area sources rather than punctual approximations.
- **Cook-Torrance PBR** with a GGX/Smith BRDF, metallic-roughness workflow, normal mapping, and USD `UsdPreviewSurface` material binding.
- **Image-based lighting** -- equirectangular-to-cubemap projection, irradiance convolution for the diffuse term, roughness-prefiltered specular mips, and a precomputed split-sum BRDF LUT.
- **Glass and transmission** -- TODO (will explore transmission LTC and spherical gaussian BTDF prefiltering, etc.)

### Shadows

- **PCSS soft shadows** -- percentage-closer soft shadows with blocker search, for distant (directional) lights as well as rect and disk area lights, so penumbra width tracks the actual emitter size.
- **Receiver-plane depth bias** -- derivative-based bias propagated through the light's clip space, which is what makes PCSS survive grazing angles without acne or peter-panning.
- **Split visibility pipeline** -- shadow visibility is generated and resolved in separate passes rather than sampled inline, decoupling shadow cost from shading and making the visibility signal available to temporal filtering.
- **Temporal resolve with variance clamping** -- history is reprojected using G-buffer motion vectors and clamped against the local neighborhood's variance, which suppresses the noise of a sparse PCSS kernel without smearing on motion.
- **Contact shadows** -- screen-space ray marching to recover the short-range occlusion that shadow-map resolution drops.

### Screen space and post

- **SSAO** with a bilateral (depth-aware) blur.
- **G-buffer motion vectors** driving history reprojection.
- **Tone mapping** as the final graph node.

## Build

```bash
pixi run build
pixi run test
```

[pixi](https://pixi.sh) brings its own Python and the whole toolchain (Conan, CMake, Ninja, clang-format), so there is nothing to bootstrap first.

## Documentation

- [Building](docs/building.md) -- prerequisites, web builds, Conan and tooling
- [Pipeline and verification](docs/pipeline.md) -- Slang/WGSL, USD assets, image-diff, headless capture
- [Known issues](docs/known-issues.md) -- open gaps and limitations, with evidence
