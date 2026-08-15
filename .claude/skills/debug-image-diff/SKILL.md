---
name: debug-image-diff
description: Investigate a failing image-diff case and propose a fix (renderer bug report / GT rebake / threshold bump). Triggers when the user asks to debug image-diff, check why a case failed, tune thresholds, or inspect FLIP heatmaps.
---

# Debugging an image-diff case

The metric is max_tile_mean FLIP over tile_size x tile_size tiles (default 64).
A pass/fail flip is almost always localized to one hot tile. Use the
worst-tile bbox from summary.json to avoid reading full-res images --
crop + stitch at the hotspot.

## Workflow

1. **Run the suspect case only.** Do not re-run all cases.

       ./repo image-diff --case <name>

2. **Read _test_captures/summary.json** for the failing case. Note:
   - score vs threshold
   - worst_tile: {x, y, w, h, mean}
   - mean_flip (whole-image reference mean; not the pass/fail metric)

3. **Side-by-side at the hotspot.** Pad the worst-tile bbox by ~32 px for
   context, crop all three images, then stitch with annotations. Example
   with bbox (X,Y,W,H) = (864,352,128,128):

       SK=.claude/skills/debug-image-diff/scripts
       ./repo python -- $SK/imgcrop.py _test_captures/<name>.png --box 864,352,128,128 -o /tmp/cap.png
       ./repo python -- $SK/imgcrop.py tests/golden/gt/<name>.png --box 864,352,128,128 -o /tmp/gt.png
       ./repo python -- $SK/imgcrop.py _test_captures/<name>.diff.png --box 864,352,128,128 -o /tmp/heat.png
       ./repo python -- $SK/imggrid.py /tmp/gt.png /tmp/cap.png /tmp/heat.png -a --labels gt,capture,flip -o /tmp/tri.png

   Then Read /tmp/tri.png.

4. **If the error is diffuse** (no single worst tile dominates): stitch
   the full-res images directly, skip cropping.

5. **Decide and act:**
   - **Renderer bug** (shadow acne, missing specular, broken SH, etc.):
     report back with a clear description of what is wrong at the hotspot.
     Do NOT silently bump the threshold.
   - **Intentional renderer change** approved by the user: rebake GT with
     ./repo bake-gt --case <name>, rerun image-diff, confirm pass.
   - **Noise / minor drift** (sub-percent score, no visual issue):
     propose a specific threshold bump in config.yaml with before/after
     scores and one line of justification.

## Script reference

- scripts/imggrid.py IMG IMG [IMG ...] -o OUT [-a] [--direction horizontal|vertical] [--labels a,b,c]
- scripts/imgcrop.py IMG --box X,Y,W,H -o OUT

Both are Pillow-only and self-contained. Invoke via ./repo python -- <script> ...
