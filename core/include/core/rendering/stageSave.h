#pragma once

#include <pxr/usd/usd/stage.h>

#include <string>

namespace pts::rendering {

/// Returns true iff the stage's root layer lives inside a `.usdz` archive.
bool is_usdz_backed(const pxr::UsdStageRefPtr& stage);

/// Save \p stage to \p out_path. Destination format is inferred from the file
/// extension (`.usda`/`.usdc`/`.usd` -> plain layer, `.usdz` -> zip archive).
///
/// Plain-layer saves preserve dependencies and rebase paths to the output
/// directory. Bundled assets remain references into their original USDZ;
/// the source package must remain available. No sibling assets are created.
/// USDZ destinations package assets into a self-contained archive.
///
/// On any per-asset extraction or export failure the function logs the error,
/// removes any partial output it created, and returns false. Never throws.
bool save_stage(const pxr::UsdStageRefPtr& stage, const std::string& out_path);

}  // namespace pts::rendering
