#pragma once

#include <pxr/usd/usd/stage.h>

#include <string>

namespace pts::rendering {

/// Returns true iff the stage's root layer lives inside a `.usdz` archive.
bool is_usdz_backed(const pxr::UsdStageRefPtr& stage);

/// Save \p stage to \p out_path. Destination format is inferred from the file
/// extension (`.usda`/`.usdc`/`.usd` -> plain layer, `.usdz` -> zip archive).
///
/// If the stage is backed by a `.usdz` package, bundled assets are extracted
/// and relocalized so the result is self-contained:
///   - For a plain-layer destination, assets are extracted next to the output
///     in a sibling `<stem>.assets/` directory (preserving the original
///     bucket subfolder structure), and the exported layer's asset paths are
///     rewritten to point at them (relative to the output file).
///   - For a `.usdz` destination, assets are packed flat alongside the root
///     layer in the new archive (not as a nested `.usdz`).
/// Non-USDZ-backed stages are written through unchanged: plain `Export()` for
/// layer destinations and `Flatten() + UsdUtilsCreateNewUsdzPackage()` for
/// `.usdz` destinations.
///
/// On any per-asset extraction or export failure the function logs the error,
/// removes any partial output it created, and returns false. Never throws.
bool save_stage(const pxr::UsdStageRefPtr& stage, const std::string& out_path);

}  // namespace pts::rendering
