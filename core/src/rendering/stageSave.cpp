#include <core/diagnostics.h>
#include <core/rendering/stageSave.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/zipFile.h>
#include <pxr/usd/usdUtils/dependencies.h>
#include <pxr/usd/usdUtils/usdzPackage.h>
#include <spdlog/spdlog.h>

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace pts::rendering {

namespace {

namespace fs = std::filesystem;

bool is_usdz_extension(const fs::path& p) {
    auto ext = p.extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext == ".usdz";
}

fs::path unique_tmp_dir(const std::string& prefix) {
    auto const ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    auto const base = fs::temp_directory_path();
    for (int i = 0; i < 32; ++i) {
        fs::path candidate = base / (prefix + std::to_string(ticks) + "_" + std::to_string(i));
        std::error_code ec;
        if (fs::create_directory(candidate, ec)) return candidate;
    }
    PANIC("Failed to allocate a unique temp directory for stage save");
}

void remove_dir(const fs::path& p) {
    std::error_code ec;
    fs::remove_all(p, ec);
}
void remove_file(const fs::path& p) {
    std::error_code ec;
    fs::remove(p, ec);
}

/// Walk every asset path in `src`, extract the referenced bytes via the asset
/// resolver (which understands `.usdz[...]` package paths), write them flat
/// into `out_dir` preserving the authored relative path (so bucket subfolders
/// from USDZ packaging survive), and return a new anonymous layer whose asset
/// paths have been rewritten to `path_prefix + authored_rel`.
/// Returns null on any extraction failure; diagnostics go through spdlog.
pxr::SdfLayerRefPtr extract_and_rewrite(const pxr::SdfLayerHandle& src, const fs::path& out_dir,
                                        const std::string& path_prefix) {
    auto work = pxr::SdfLayer::CreateAnonymous(
        fs::path(src->GetIdentifier()).filename().replace_extension(".usda").string());
    INVARIANT_MSG(work, "SdfLayer::CreateAnonymous returned null");
    work->TransferContent(src);

    std::vector<std::string> errors;
    std::unordered_map<std::string, std::string> remap;

    pxr::UsdUtilsModifyAssetPaths(work, [&](const std::string& authored) -> std::string {
        if (authored.empty()) return authored;
        if (auto it = remap.find(authored); it != remap.end()) return it->second;

        fs::path const rel = fs::u8path(authored).lexically_normal();
        if (rel.empty() || rel.is_absolute()) {
            errors.push_back("cannot relocalize non-relative path '" + authored + "'");
            return authored;
        }

        std::string const abs_ref = src->ComputeAbsolutePath(authored);
        auto asset = pxr::ArGetResolver().OpenAsset(pxr::ArResolvedPath(abs_ref));
        if (!asset) {
            errors.push_back("OpenAsset failed for '" + authored + "' (resolved '" + abs_ref +
                             "')");
            return authored;
        }

        fs::path const target = out_dir / rel;
        std::error_code ec;
        fs::create_directories(target.parent_path(), ec);
        if (ec) {
            errors.push_back("mkdir '" + target.parent_path().string() + "': " + ec.message());
            return authored;
        }
        auto buffer = asset->GetBuffer();
        if (!buffer) {
            errors.push_back("asset '" + authored + "' has null buffer");
            return authored;
        }
        std::ofstream out(target, std::ios::binary);
        out.write(buffer.get(), static_cast<std::streamsize>(asset->GetSize()));
        out.close();
        if (!out) {
            errors.push_back("write failed: " + target.string());
            return authored;
        }

        std::string new_path = path_prefix.empty() ? rel.generic_string()
                                                   : (fs::path(path_prefix) / rel).generic_string();
        remap.emplace(authored, std::move(new_path));
        return remap.at(authored);
    });

    if (!errors.empty()) {
        for (auto const& e : errors) spdlog::error("[save_stage] {}", e);
        return {};
    }
    return work;
}

/// Non-USDZ-backed + .usdz destination: flatten and package. The flattened
/// layer's references resolve from the source stage's context, so
/// `UsdUtilsCreateNewUsdzPackage` produces a clean flat archive.
bool export_nonusdz_to_usdz(const pxr::UsdStageRefPtr& stage, const std::string& out_path) {
    auto flat = stage->Flatten();
    if (!flat) {
        spdlog::error("[save_stage] Flatten() failed for {}", out_path);
        return false;
    }
    fs::path const tmp = unique_tmp_dir("_pts_save_") / "export.usda";
    std::error_code ec;
    fs::create_directories(tmp.parent_path(), ec);
    if (!flat->Export(tmp.string())) {
        spdlog::error("[save_stage] Flattened Export failed: {}", tmp.string());
        remove_dir(tmp.parent_path());
        return false;
    }
    bool const ok = pxr::UsdUtilsCreateNewUsdzPackage(pxr::SdfAssetPath(tmp.string()), out_path);
    remove_dir(tmp.parent_path());
    if (!ok) {
        spdlog::error("[save_stage] UsdUtilsCreateNewUsdzPackage failed: {}", out_path);
        remove_file(fs::u8path(out_path));
        return false;
    }
    return true;
}

}  // namespace

bool is_usdz_backed(const pxr::UsdStageRefPtr& stage) {
    if (!stage) return false;
    auto layer = stage->GetRootLayer();
    if (!layer) return false;
    auto format = layer->GetFileFormat();
    return format && format->IsPackage();
}

bool save_stage(const pxr::UsdStageRefPtr& stage, const std::string& out_path) {
    PRECONDITION(stage);
    PRECONDITION(!out_path.empty());

    fs::path const out = fs::u8path(out_path);
    bool const dest_is_usdz = is_usdz_extension(out);

    if (!is_usdz_backed(stage)) {
        if (dest_is_usdz) return export_nonusdz_to_usdz(stage, out_path);
        if (!stage->GetRootLayer()->Export(out_path)) {
            spdlog::error("[save_stage] GetRootLayer()->Export failed: {}", out_path);
            return false;
        }
        return true;
    }

    // USDZ-backed: extract bundled assets individually (USD has no one-liner
    // for this; ArResolver::OpenAsset is the primitive that understands
    // `.usdz[...]` package paths), rewrite the root layer's paths to point at
    // the extracted files, then write the destination format.
    std::string const stem = out.stem().string();
    fs::path const assets_dir =
        dest_is_usdz ? unique_tmp_dir("_pts_pack_") : (out.parent_path() / (stem + ".assets"));
    std::string const path_prefix = dest_is_usdz ? std::string{} : (stem + ".assets");

    std::error_code ec;
    fs::create_directories(assets_dir, ec);
    if (ec) {
        spdlog::error("[save_stage] mkdir {}: {}", assets_dir.string(), ec.message());
        if (dest_is_usdz) remove_dir(assets_dir);
        return false;
    }

    auto work = extract_and_rewrite(stage->GetRootLayer(), assets_dir, path_prefix);
    if (!work) {
        remove_dir(assets_dir);
        return false;
    }

    if (!dest_is_usdz) {
        if (!work->Export(out_path)) {
            spdlog::error("[save_stage] Export rewritten layer failed: {}", out_path);
            remove_dir(assets_dir);
            return false;
        }
        return true;
    }

    // .usdz destination: zip the extracted assets + rewritten root layer.
    fs::path const layer_path = assets_dir / (stem + ".usda");
    if (!work->Export(layer_path.string())) {
        spdlog::error("[save_stage] Export rewritten layer failed: {}", layer_path.string());
        remove_dir(assets_dir);
        return false;
    }
    auto writer = pxr::SdfZipFileWriter::CreateNew(out_path);
    if (!writer) {
        spdlog::error("[save_stage] SdfZipFileWriter::CreateNew failed: {}", out_path);
        remove_dir(assets_dir);
        return false;
    }
    // Root layer must be first entry per USDZ convention.
    if (writer.AddFile(layer_path.string(), layer_path.filename().string()).empty()) {
        spdlog::error("[save_stage] Failed to add root layer to archive");
        writer.Discard();
        remove_dir(assets_dir);
        return false;
    }
    for (auto const& entry : fs::recursive_directory_iterator(assets_dir)) {
        if (!entry.is_regular_file() || entry.path() == layer_path) continue;
        auto const rel = fs::relative(entry.path(), assets_dir).generic_string();
        if (writer.AddFile(entry.path().string(), rel).empty()) {
            spdlog::error("[save_stage] Failed to add {} to archive", rel);
            writer.Discard();
            remove_dir(assets_dir);
            return false;
        }
    }
    if (!writer.Save()) {
        spdlog::error("[save_stage] SdfZipFileWriter::Save failed: {}", out_path);
        remove_dir(assets_dir);
        remove_file(out);
        return false;
    }
    remove_dir(assets_dir);
    return true;
}

}  // namespace pts::rendering
