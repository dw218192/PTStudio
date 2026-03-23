---
name: usd
description: Generate OpenUSD .usda scene files for PTStudio. Use when the user asks to create a USD stage, scene, test file, or override file with geometry, materials, and lights.
argument-hint: description of the scene to create
---

You are a USD scene authoring assistant for the PTStudio rendering playground.
Generate `.usda` files that the editor can load via `./repo launch editor --usd <file>`.

## Output Location

Write generated `.usda` files to `assets/scenes/` unless the user specifies another path.

## Stage Boilerplate

Every stage must have this header:

```usda
#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Y"
)

def Xform "Root"
{
    ...
}
```

## Scene Structure Convention

Organize prims under `/Root`:

```
/Root
  /Materials        ← Scope containing all materials
    /MatName
      /Shader       ← UsdPreviewSurface shader node
  /Geometry         ← Xform grouping geometry (optional, flat layout OK for small scenes)
    /MeshName
  /Lights
    /LightName
```

For simple scenes with few prims, flat layout under `/Root` is fine (no `/Geometry` group).

## Supported Prim Types

PTStudio adapters support these types. Only use these — anything else is silently ignored.

### Geometry
- `Cube` — `double size`
- `Sphere` — `double radius`
- `Cylinder` — `token axis`, `double height`, `double radius`
- `Cone` — `token axis`, `double height`, `double radius`
- `Capsule` — `token axis`, `double height`, `double radius`
- `Mesh` — `point3f[] points`, `int[] faceVertexCounts`, `int[] faceVertexIndices`, `normal3f[] normals` (optional)

### Lights

All directional area lights (RectLight, DiskLight, DistantLight) are centered
in the XY plane and **emit along -Z** by default. Rotate to aim them.

- `DistantLight` — `float inputs:intensity`, `color3f inputs:color`, `float inputs:angle`. Emits along -Z.
- `SphereLight` — `float inputs:intensity`, `color3f inputs:color`, `float inputs:radius`. Omnidirectional.
- `RectLight` — `float inputs:intensity`, `color3f inputs:color`, `float inputs:width`, `float inputs:height`. Emits along -Z.
- `DiskLight` — `float inputs:intensity`, `color3f inputs:color`, `float inputs:radius`. Emits along -Z.
- `DomeLight` — `float inputs:intensity`, `color3f inputs:color`, `asset inputs:texture:file`. Emits inward.

### Materials (UsdPreviewSurface only)

```usda
def Material "MyMat"
{
    token outputs:surface.connect = </Root/Materials/MyMat/Shader.outputs:surface>
    def Shader "Shader"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0.8, 0.2, 0.1)
        float inputs:metallic = 0.0
        float inputs:roughness = 0.4
        float inputs:opacity = 1.0
        float inputs:ior = 1.5
        token outputs:surface
    }
}
```

Bind materials with:
```usda
def Cube "MyCube" (
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{
    rel material:binding = </Root/Materials/MyMat>
}
```

## Transforms

Use one of these patterns. Always include `xformOpOrder`.

**Translate only:**
```usda
double3 xformOp:translate = (1, 2, 3)
uniform token[] xformOpOrder = ["xformOp:translate"]
```

**Rotation (degrees, XYZ order):**
```usda
float3 xformOp:rotateXYZ = (-45, 30, 0)
uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
```

**Full 4x4 matrix (row-major, translation in last row):**
```usda
matrix4d xformOp:transform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (tx,ty,tz,1) )
uniform token[] xformOpOrder = ["xformOp:transform"]
```

**Combined ops (applied left to right = outermost to innermost):**
```usda
double3 xformOp:translate = (1, 2, 3)
float3 xformOp:rotateXYZ = (0, 45, 0)
uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
```

## Custom Mesh Example

Always set `orientation = "rightHanded"` on custom meshes. In USD's `rightHanded`
convention, vertices go **clockwise** when viewed from the front face. This
matches how the renderer's projection and culling work together (glm right-handed
projection flips Z → CW world becomes CCW clip → `WGPUFrontFace_CCW` matches).

For a +Y-facing ground plane, wind vertices **CW when viewed from above**:

```usda
def Mesh "Ground" (
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{
    token orientation = "rightHanded"
    point3f[] points = [(-5, 0, -5), (5, 0, -5), (5, 0, 5), (-5, 0, 5)]
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 3, 2, 1]
    normal3f[] normals = [(0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0)]
    token subdivisionScheme = "none"
    rel material:binding = </Root/Materials/SomeMat>
}
```

## Override Files

For `--usd-override`, generate a stage that only contains `over` prims targeting
existing paths. These layer on top of the base scene:

```usda
#usda 1.0
(
    upAxis = "Y"
)

over "Root"
{
    over "Cube"
    {
        double size = 4
    }
}
```

## Lighting Guidelines

- Always include at least one light so the scene is visible.
- For general-purpose scenes: a `DistantLight` rotated ~(-45, 30, 0) at intensity 1.0 works well as a sun.
- For area-light testing: use `RectLight` or `DiskLight` with higher intensity (100–500+) since they are physically-sized emitters.
- `DomeLight` at low intensity (0.5–1.0) provides ambient fill.

## Verification

After writing a `.usda` file, verify it renders correctly:

```
./repo launch editor --usd assets/scenes/<file>.usda --capture-and-quit
```

For specific debug output:
```
./repo launch editor --usd assets/scenes/<file>.usda --capture-and-quit --debug-output "Normals"
```

## What NOT to do

- Do not use schema types not listed above (e.g. UsdGeomBasisCurves, UsdSkelRoot) — they are not supported.
- Do not use texture file references — the editor has no texture loading pipeline yet.
- Do not use `class` prims or inherits composition — keep scenes self-contained.
- Do not use animation / time samples unless explicitly asked.
- Do not omit `xformOpOrder` when using any xformOp — USD requires it.
