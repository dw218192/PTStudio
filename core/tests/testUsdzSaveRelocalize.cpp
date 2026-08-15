#include <core/rendering/stageSave.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/zipFile.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdUtils/usdzPackage.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "testApplication.h"

namespace fs = std::filesystem;

namespace {

/// Write a trivial 2x2 PPM image. Parsed by stbi_load but also easy to
/// verify byte-for-byte in tests.
void write_ppm(const fs::path& path, std::uint8_t fill) {
    std::ofstream f(path, std::ios::binary);
    REQUIRE(f);
    f << "P6\n2 2\n255\n";
    for (int i = 0; i < 4; ++i) {
        f.put(static_cast<char>(fill));
        f.put(static_cast<char>(fill));
        f.put(static_cast<char>(fill));
    }
    f.close();
    REQUIRE(f);
}

/// Build a USDA stage on disk that references two external textures via
/// UsdPreviewSurface shaders. Returns the written .usda path.
fs::path build_source_usda(const fs::path& dir) {
    fs::create_directories(dir / "tex");
    auto tex_a = dir / "tex" / "a.png";
    auto tex_b = dir / "tex" / "b.png";
    write_ppm(tex_a, 0x11);
    write_ppm(tex_b, 0xAA);

    auto usda_path = dir / "scene.usda";
    auto layer = pxr::SdfLayer::CreateNew(usda_path.string());
    REQUIRE(layer);
    auto stage = pxr::UsdStage::Open(layer);
    REQUIRE(stage);

    pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));

    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/Mesh"));
    pxr::VtVec3fArray points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.GetPointsAttr().Set(points);
    mesh.GetFaceVertexCountsAttr().Set(pxr::VtIntArray{3});
    mesh.GetFaceVertexIndicesAttr().Set(pxr::VtIntArray{0, 1, 2});

    auto material = pxr::UsdShadeMaterial::Define(stage, pxr::SdfPath("/Root/Mat"));
    auto surface = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Root/Mat/Surface"));
    surface.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdPreviewSurface")));

    auto diffuse = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Root/Mat/Diffuse"));
    diffuse.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdUVTexture")));
    diffuse.CreateInput(pxr::TfToken("file"), pxr::SdfValueTypeNames->Asset)
        .Set(pxr::SdfAssetPath("tex/a.png"));

    auto normal = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Root/Mat/Normal"));
    normal.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdUVTexture")));
    normal.CreateInput(pxr::TfToken("file"), pxr::SdfValueTypeNames->Asset)
        .Set(pxr::SdfAssetPath("tex/b.png"));

    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(),
                                                   pxr::TfToken("surface"));
    pxr::UsdShadeMaterialBindingAPI::Apply(mesh.GetPrim()).Bind(material);

    stage->Save();
    return usda_path;
}

/// Package the source .usda (and its texture dependencies) into a .usdz via
/// UsdUtilsCreateNewUsdzPackage, which is what PTStudio's build pipeline
/// does -- bucketing bundled assets into numeric subfolders.
fs::path package_to_usdz(const fs::path& usda, const fs::path& out_dir) {
    auto usdz = out_dir / "packaged.usdz";
    bool const ok =
        pxr::UsdUtilsCreateNewUsdzPackage(pxr::SdfAssetPath(usda.string()), usdz.string());
    REQUIRE(ok);
    REQUIRE(fs::exists(usdz));
    return usdz;
}

std::vector<std::string> collect_asset_paths(const pxr::UsdStageRefPtr& stage) {
    std::vector<std::string> out;
    for (auto prim : stage->TraverseAll()) {
        for (auto const& attr : prim.GetAuthoredAttributes()) {
            if (attr.GetTypeName() != pxr::SdfValueTypeNames->Asset) continue;
            pxr::SdfAssetPath v;
            if (!attr.Get(&v)) continue;
            out.push_back(v.GetAssetPath());
        }
    }
    return out;
}

bool path_resolves_to_file_on_disk(const pxr::UsdStageRefPtr& stage) {
    for (auto prim : stage->TraverseAll()) {
        for (auto const& attr : prim.GetAuthoredAttributes()) {
            if (attr.GetTypeName() != pxr::SdfValueTypeNames->Asset) continue;
            pxr::SdfAssetPath v;
            if (!attr.Get(&v)) continue;
            if (v.GetAssetPath().empty()) continue;
            auto const& resolved = v.GetResolvedPath();
            if (resolved.empty()) return false;
            auto asset = pxr::ArGetResolver().OpenAsset(pxr::ArResolvedPath(resolved));
            if (!asset) return false;
        }
    }
    return true;
}

fs::path make_scratch_dir(const std::string& label) {
    auto base = fs::temp_directory_path() /
                ("pts_relocalize_test_" + label + "_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(base);
    return base;
}

}  // namespace

TEST_CASE("save_stage relocalizes USDZ-backed stage to .usda with sibling assets") {
    auto scratch = make_scratch_dir("usda");
    auto src_usda = build_source_usda(scratch / "src");
    auto src_usdz = package_to_usdz(src_usda, scratch);

    auto src_stage = pxr::UsdStage::Open(src_usdz.string());
    REQUIRE(src_stage);
    REQUIRE(pts::rendering::is_usdz_backed(src_stage));

    auto out_dir = scratch / "out";
    fs::create_directories(out_dir);
    auto out_usda = out_dir / "scene.usda";

    REQUIRE(pts::rendering::save_stage(src_stage, out_usda.string()));
    REQUIRE(fs::exists(out_usda));

    auto assets_dir = out_dir / "scene.assets";
    CHECK(fs::exists(assets_dir));

    // At least one bucket subdirectory should exist (usdzPackage buckets
    // into numeric folders).
    bool has_bucket_file = false;
    for (auto const& entry : fs::recursive_directory_iterator(assets_dir)) {
        if (entry.is_regular_file()) has_bucket_file = true;
    }
    CHECK(has_bucket_file);

    // Re-open the saved .usda and confirm every asset path resolves to an
    // on-disk file via the default ArResolver.
    auto out_stage = pxr::UsdStage::Open(out_usda.string());
    REQUIRE(out_stage);
    CHECK(!pts::rendering::is_usdz_backed(out_stage));
    CHECK(path_resolves_to_file_on_disk(out_stage));

    // Authored paths in the saved layer should reference the sibling
    // <stem>.assets/ directory, not the original bundle layout.
    for (auto const& p : collect_asset_paths(out_stage)) {
        CHECK(p.find("scene.assets/") == 0);
    }

    std::error_code ec;
    fs::remove_all(scratch, ec);
}

TEST_CASE("save_stage relocalizes USDZ-backed stage to .usdz without nesting") {
    auto scratch = make_scratch_dir("usdz");
    auto src_usda = build_source_usda(scratch / "src");
    auto src_usdz = package_to_usdz(src_usda, scratch);

    auto src_stage = pxr::UsdStage::Open(src_usdz.string());
    REQUIRE(src_stage);

    auto out_dir = scratch / "out";
    fs::create_directories(out_dir);
    auto out_usdz = out_dir / "scene.usdz";

    REQUIRE(pts::rendering::save_stage(src_stage, out_usdz.string()));
    REQUIRE(fs::exists(out_usdz));

    // Reopen and confirm assets still resolve.
    auto out_stage = pxr::UsdStage::Open(out_usdz.string());
    REQUIRE(out_stage);
    CHECK(path_resolves_to_file_on_disk(out_stage));

    // Inspect archive entries: confirm no nested .usdz is present (the exact
    // bug the relocalize helper avoids). Every entry should be a leaf
    // asset, not another package.
    auto zip = pxr::SdfZipFile::Open(out_usdz.string());
    REQUIRE(bool(zip));
    bool has_nested_usdz = false;
    std::size_t entry_count = 0;
    for (auto it = zip.begin(); it != zip.end(); ++it) {
        ++entry_count;
        std::string const name = *it;
        auto const dot = name.rfind('.');
        if (dot != std::string::npos && name.substr(dot) == ".usdz") {
            has_nested_usdz = true;
        }
    }
    CHECK(entry_count > 0);
    CHECK_FALSE(has_nested_usdz);

    std::error_code ec;
    fs::remove_all(scratch, ec);
}

TEST_CASE("save_stage leaves non-USDZ-backed stages unchanged") {
    auto scratch = make_scratch_dir("noop");
    auto src_usda = build_source_usda(scratch / "src");

    auto src_stage = pxr::UsdStage::Open(src_usda.string());
    REQUIRE(src_stage);
    CHECK_FALSE(pts::rendering::is_usdz_backed(src_stage));

    auto out_usda = scratch / "out.usda";
    REQUIRE(pts::rendering::save_stage(src_stage, out_usda.string()));
    REQUIRE(fs::exists(out_usda));

    // No sibling assets/ directory is produced for a plain-layer stage.
    CHECK_FALSE(fs::exists(scratch / "out.assets"));

    // Output should round-trip via Export: compare against a direct
    // GetRootLayer()->Export reference.
    auto reference = scratch / "ref.usda";
    REQUIRE(src_stage->GetRootLayer()->Export(reference.string()));

    auto read_all = [](const fs::path& p) {
        std::ifstream in(p, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    CHECK(read_all(out_usda) == read_all(reference));

    std::error_code ec;
    fs::remove_all(scratch, ec);
}

TEST_CASE("save_stage fails loudly and leaves no partial output on extraction error") {
    auto scratch = make_scratch_dir("err");
    auto src_usda = build_source_usda(scratch / "src");
    auto src_usdz = package_to_usdz(src_usda, scratch);

    // Corrupt the packaged .usdz so one referenced asset cannot be opened:
    // hand-rebuild the archive with a mangled inner layer that references
    // "missing.png", but omit that file. The default resolver will fail
    // OpenAsset, so save_stage must report failure and clean up.
    auto corrupted = scratch / "corrupted.usdz";
    {
        auto broken_usda = scratch / "broken.usda";
        auto layer = pxr::SdfLayer::CreateNew(broken_usda.string());
        REQUIRE(layer);
        auto stage = pxr::UsdStage::Open(layer);
        REQUIRE(stage);
        pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
        auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/Mesh"));
        pxr::VtVec3fArray points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        mesh.GetPointsAttr().Set(points);
        mesh.GetFaceVertexCountsAttr().Set(pxr::VtIntArray{3});
        mesh.GetFaceVertexIndicesAttr().Set(pxr::VtIntArray{0, 1, 2});
        auto material = pxr::UsdShadeMaterial::Define(stage, pxr::SdfPath("/Root/Mat"));
        auto surf = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Root/Mat/Tex"));
        surf.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdUVTexture")));
        // Reference a path that will live inside bucket "1/" but we won't
        // actually pack its file below -- so OpenAsset will fail.
        surf.CreateInput(pxr::TfToken("file"), pxr::SdfValueTypeNames->Asset)
            .Set(pxr::SdfAssetPath("1/missing.png"));
        stage->Save();

        auto writer = pxr::SdfZipFileWriter::CreateNew(corrupted.string());
        REQUIRE(bool(writer));
        REQUIRE(!writer.AddFile(broken_usda.string(), "broken.usda").empty());
        REQUIRE(writer.Save());
    }

    auto src_stage = pxr::UsdStage::Open(corrupted.string());
    REQUIRE(src_stage);
    REQUIRE(pts::rendering::is_usdz_backed(src_stage));

    auto out_dir = scratch / "out";
    fs::create_directories(out_dir);
    auto out_usda = out_dir / "scene.usda";

    CHECK_FALSE(pts::rendering::save_stage(src_stage, out_usda.string()));
    CHECK_FALSE(fs::exists(out_usda));

    std::error_code ec;
    fs::remove_all(scratch, ec);
}

PTS_TEST_MAIN()
