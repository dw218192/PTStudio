#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfOutputFile.h>
#include <core/rendering/adapterHelpers.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "testApplication.h"

namespace {

/// Create a minimal 2x2 PPM image file that stbi_load can read.
/// Returns the absolute path. Caller must remove the file when done.
std::string create_test_texture(const std::string& filename) {
    // Use the system temp directory
#ifdef _WIN32
    const char* tmp = std::getenv("TEMP");
    if (!tmp) tmp = std::getenv("TMP");
    if (!tmp) tmp = ".";
#else
    const char* tmp = "/tmp";
#endif
    std::string path = std::string(tmp) + "/" + filename;
    std::ofstream f(path, std::ios::binary);
    // PPM P6 format: 2x2 RGB image
    f << "P6\n2 2\n255\n";
    // 4 pixels x 3 channels = 12 bytes of pixel data
    for (int i = 0; i < 4; ++i) {
        f.put(static_cast<char>(255));
        f.put(static_cast<char>(0));
        f.put(static_cast<char>(255));
    }
    f.close();
    return path;
}

void remove_test_texture(const std::string& path) {
    std::remove(path.c_str());
}

std::string create_test_exr(const std::string& filename) {
#ifdef _WIN32
    const char* tmp = std::getenv("TEMP");
    if (!tmp) tmp = std::getenv("TMP");
    if (!tmp) tmp = ".";
#else
    const char* tmp = "/tmp";
#endif
    std::string path = std::string(tmp) + "/" + filename;
    // 2x2 RGBA float pixels: red, green, blue, white
    float pixels[16] = {1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1};
    Imf::Header header(2, 2);
    header.channels().insert("R", Imf::Channel(Imf::FLOAT));
    header.channels().insert("G", Imf::Channel(Imf::FLOAT));
    header.channels().insert("B", Imf::Channel(Imf::FLOAT));
    header.channels().insert("A", Imf::Channel(Imf::FLOAT));
    Imf::OutputFile file(path.c_str(), header);
    Imf::FrameBuffer fb;
    size_t xs = 4 * sizeof(float), ys = 2 * xs;
    fb.insert("R", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&pixels[0]), xs, ys));
    fb.insert("G", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&pixels[1]), xs, ys));
    fb.insert("B", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&pixels[2]), xs, ys));
    fb.insert("A", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&pixels[3]), xs, ys));
    file.setFrameBuffer(fb);
    file.writePixels(2);
    return path;
}

/// Build a minimal USD stage with a mesh, material, and UsdUVTexture nodes.
/// texture_path: absolute path to an image file for the diffuse texture.
pxr::UsdStageRefPtr create_textured_stage(const std::string& texture_path) {
    auto stage = pxr::UsdStage::CreateInMemory();

    pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));

    // Mesh with minimal geometry
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/Mesh"));
    mesh.GetPointsAttr().Set(pxr::VtVec3fArray{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
    mesh.GetFaceVertexCountsAttr().Set(pxr::VtIntArray{3});
    mesh.GetFaceVertexIndicesAttr().Set(pxr::VtIntArray{0, 1, 2});

    // Material
    auto material = pxr::UsdShadeMaterial::Define(stage, pxr::SdfPath("/Root/Mat"));
    auto surface = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Root/Mat/Surface"));
    surface.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdPreviewSurface")));

    // Scalar fallback values
    surface.CreateInput(pxr::TfToken("diffuseColor"), pxr::SdfValueTypeNames->Color3f)
        .Set(pxr::GfVec3f(0.8f, 0.2f, 0.1f));
    surface.CreateInput(pxr::TfToken("metallic"), pxr::SdfValueTypeNames->Float).Set(0.9f);
    surface.CreateInput(pxr::TfToken("roughness"), pxr::SdfValueTypeNames->Float).Set(0.3f);
    surface.CreateInput(pxr::TfToken("opacity"), pxr::SdfValueTypeNames->Float).Set(0.7f);
    surface.CreateInput(pxr::TfToken("ior"), pxr::SdfValueTypeNames->Float).Set(1.45f);
    surface.CreateInput(pxr::TfToken("emissiveColor"), pxr::SdfValueTypeNames->Color3f)
        .Set(pxr::GfVec3f(0.1f, 0.2f, 0.3f));

    // UsdUVTexture for diffuseColor
    auto diffuse_tex = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Root/Mat/DiffuseTex"));
    diffuse_tex.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdUVTexture")));
    diffuse_tex.CreateInput(pxr::TfToken("file"), pxr::SdfValueTypeNames->Asset)
        .Set(pxr::SdfAssetPath(texture_path));
    diffuse_tex.CreateOutput(pxr::TfToken("rgb"), pxr::SdfValueTypeNames->Float3);

    // Connect diffuseColor -> DiffuseTex.rgb
    surface.GetInput(pxr::TfToken("diffuseColor"))
        .ConnectToSource(diffuse_tex.ConnectableAPI(), pxr::TfToken("rgb"));

    // UsdUVTexture for metallic (shared ORM texture, reading specific channel)
    auto orm_tex = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Root/Mat/OrmTex"));
    orm_tex.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdUVTexture")));
    orm_tex.CreateInput(pxr::TfToken("file"), pxr::SdfValueTypeNames->Asset)
        .Set(pxr::SdfAssetPath(texture_path));
    orm_tex.CreateOutput(pxr::TfToken("r"), pxr::SdfValueTypeNames->Float);

    // Connect metallic -> OrmTex.r
    surface.GetInput(pxr::TfToken("metallic"))
        .ConnectToSource(orm_tex.ConnectableAPI(), pxr::TfToken("r"));

    // Connect roughness -> OrmTex.g
    orm_tex.CreateOutput(pxr::TfToken("g"), pxr::SdfValueTypeNames->Float);
    surface.GetInput(pxr::TfToken("roughness"))
        .ConnectToSource(orm_tex.ConnectableAPI(), pxr::TfToken("g"));

    // Connect opacity -> OrmTex.a
    orm_tex.CreateOutput(pxr::TfToken("a"), pxr::SdfValueTypeNames->Float);
    surface.GetInput(pxr::TfToken("opacity"))
        .ConnectToSource(orm_tex.ConnectableAPI(), pxr::TfToken("a"));

    // UsdUVTexture for normal
    auto normal_tex = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Root/Mat/NormalTex"));
    normal_tex.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdUVTexture")));
    normal_tex.CreateInput(pxr::TfToken("file"), pxr::SdfValueTypeNames->Asset)
        .Set(pxr::SdfAssetPath(texture_path));
    normal_tex.CreateOutput(pxr::TfToken("rgb"), pxr::SdfValueTypeNames->Float3);

    surface.CreateInput(pxr::TfToken("normal"), pxr::SdfValueTypeNames->Normal3f);
    surface.GetInput(pxr::TfToken("normal"))
        .ConnectToSource(normal_tex.ConnectableAPI(), pxr::TfToken("rgb"));

    // UsdUVTexture for emissiveColor
    auto emissive_tex = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Root/Mat/EmissiveTex"));
    emissive_tex.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdUVTexture")));
    emissive_tex.CreateInput(pxr::TfToken("file"), pxr::SdfValueTypeNames->Asset)
        .Set(pxr::SdfAssetPath(texture_path));
    emissive_tex.CreateOutput(pxr::TfToken("rgb"), pxr::SdfValueTypeNames->Float3);

    surface.GetInput(pxr::TfToken("emissiveColor"))
        .ConnectToSource(emissive_tex.ConnectableAPI(), pxr::TfToken("rgb"));

    // Wire surface output
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(),
                                                   pxr::TfToken("surface"));

    // Bind material to mesh
    pxr::UsdShadeMaterialBindingAPI::Apply(mesh.GetPrim()).Bind(material);

    return stage;
}

}  // namespace

// --- Tests that don't need GPU (run on all platforms) ---

TEST_CASE("read_preview_surface reads scalar values") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto material = pxr::UsdShadeMaterial::Define(stage, pxr::SdfPath("/Mat"));
    auto surface = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Mat/Surface"));
    surface.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdPreviewSurface")));
    surface.CreateInput(pxr::TfToken("diffuseColor"), pxr::SdfValueTypeNames->Color3f)
        .Set(pxr::GfVec3f(0.8f, 0.2f, 0.1f));
    surface.CreateInput(pxr::TfToken("metallic"), pxr::SdfValueTypeNames->Float).Set(0.9f);
    surface.CreateInput(pxr::TfToken("roughness"), pxr::SdfValueTypeNames->Float).Set(0.3f);
    surface.CreateInput(pxr::TfToken("opacity"), pxr::SdfValueTypeNames->Float).Set(0.7f);
    surface.CreateInput(pxr::TfToken("emissiveColor"), pxr::SdfValueTypeNames->Color3f)
        .Set(pxr::GfVec3f(0.1f, 0.2f, 0.3f));

    pts::rendering::RenderWorld world;
    auto scope = world.begin_sync();
    auto mat = pts::rendering::read_preview_surface(surface, scope);

    CHECK(mat.diffuse_color.x == doctest::Approx(0.8f));
    CHECK(mat.diffuse_color.y == doctest::Approx(0.2f));
    CHECK(mat.diffuse_color.z == doctest::Approx(0.1f));
    CHECK(mat.metallic == doctest::Approx(0.9f));
    CHECK(mat.roughness == doctest::Approx(0.3f));
    CHECK(mat.opacity == doctest::Approx(0.7f));
    CHECK(mat.emissive_color.x == doctest::Approx(0.1f));
    CHECK(mat.emissive_color.y == doctest::Approx(0.2f));
    CHECK(mat.emissive_color.z == doctest::Approx(0.3f));
    // No textures connected -- all should be UINT32_MAX
    CHECK(mat.diffuse_tex == UINT32_MAX);
    CHECK(mat.normal_tex == UINT32_MAX);
    CHECK(mat.metallic_tex == UINT32_MAX);
    CHECK(mat.roughness_tex == UINT32_MAX);
    CHECK(mat.emissive_tex == UINT32_MAX);
    CHECK(mat.opacity_tex == UINT32_MAX);
    CHECK(mat.ior == doctest::Approx(1.5f));
}

TEST_CASE("read_preview_surface walks texture connections") {
    auto tex_path = create_test_texture("pts_test_diffuse.ppm");

    auto stage = create_textured_stage(tex_path);
    auto surface = pxr::UsdShadeShader::Get(stage, pxr::SdfPath("/Root/Mat/Surface"));
    REQUIRE(bool(surface));

    pts::rendering::RenderWorld world;
    auto scope = world.begin_sync();
    auto mat = pts::rendering::read_preview_surface(surface, scope);

    // All 6 inputs have texture connections with valid files
    CHECK(mat.diffuse_tex != UINT32_MAX);
    CHECK(mat.metallic_tex != UINT32_MAX);
    CHECK(mat.roughness_tex != UINT32_MAX);
    CHECK(mat.opacity_tex != UINT32_MAX);
    CHECK(mat.normal_tex != UINT32_MAX);
    CHECK(mat.emissive_tex != UINT32_MAX);

    // Scalar fallback values should still be set
    CHECK(mat.diffuse_color.x == doctest::Approx(0.8f));
    CHECK(mat.metallic == doctest::Approx(0.9f));
    CHECK(mat.roughness == doctest::Approx(0.3f));
    CHECK(mat.opacity == doctest::Approx(0.7f));

    remove_test_texture(tex_path);
}

TEST_CASE("read_preview_surface reads ior from UsdPreviewSurface") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto material = pxr::UsdShadeMaterial::Define(stage, pxr::SdfPath("/Mat"));
    auto surface = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Mat/Surface"));
    surface.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdPreviewSurface")));
    surface.CreateInput(pxr::TfToken("ior"), pxr::SdfValueTypeNames->Float).Set(1.45f);

    pts::rendering::RenderWorld world;
    auto scope = world.begin_sync();
    auto mat = pts::rendering::read_preview_surface(surface, scope);

    CHECK(mat.ior == doctest::Approx(1.45f));
}

TEST_CASE("read_preview_surface defaults ior to 1.5 when not specified") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto material = pxr::UsdShadeMaterial::Define(stage, pxr::SdfPath("/Mat"));
    auto surface = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Mat/Surface"));
    surface.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdPreviewSurface")));

    pts::rendering::RenderWorld world;
    auto scope = world.begin_sync();
    auto mat = pts::rendering::read_preview_surface(surface, scope);

    CHECK(mat.ior == doctest::Approx(1.5f));
}

TEST_CASE("read_preview_surface with unresolvable texture keeps UINT32_MAX") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto material = pxr::UsdShadeMaterial::Define(stage, pxr::SdfPath("/Mat"));
    auto surface = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Mat/Surface"));
    surface.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdPreviewSurface")));
    surface.CreateInput(pxr::TfToken("diffuseColor"), pxr::SdfValueTypeNames->Color3f)
        .Set(pxr::GfVec3f(0.5f, 0.5f, 0.5f));

    // Connect to a UsdUVTexture with a nonexistent file
    auto tex = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Mat/BadTex"));
    tex.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdUVTexture")));
    tex.CreateInput(pxr::TfToken("file"), pxr::SdfValueTypeNames->Asset)
        .Set(pxr::SdfAssetPath("/nonexistent/texture.png"));
    tex.CreateOutput(pxr::TfToken("rgb"), pxr::SdfValueTypeNames->Float3);
    surface.GetInput(pxr::TfToken("diffuseColor"))
        .ConnectToSource(tex.ConnectableAPI(), pxr::TfToken("rgb"));

    pts::rendering::RenderWorld world;
    auto scope = world.begin_sync();
    auto mat = pts::rendering::read_preview_surface(surface, scope);

    // Texture file doesn't exist -- should remain UINT32_MAX
    CHECK(mat.diffuse_tex == UINT32_MAX);
    // Scalar value should still be read
    CHECK(mat.diffuse_color.x == doctest::Approx(0.5f));
}

TEST_CASE("resolve_material with textures populates material correctly") {
    auto tex_path = create_test_texture("pts_test_resolve.ppm");
    auto stage = create_textured_stage(tex_path);

    pts::rendering::RenderWorld world;
    auto scope = world.begin_sync();
    auto prim = stage->GetPrimAtPath(pxr::SdfPath("/Root/Mesh"));
    REQUIRE(prim.IsValid());

    auto idx = pts::rendering::resolve_material(prim, scope);
    REQUIRE(idx != pts::rendering::k_no_material);

    auto& mat = scope.materials()[idx];
    CHECK(mat.diffuse_tex != UINT32_MAX);
    CHECK(mat.metallic_tex != UINT32_MAX);
    CHECK(mat.roughness_tex != UINT32_MAX);
    CHECK(mat.opacity_tex != UINT32_MAX);
    CHECK(mat.normal_tex != UINT32_MAX);
    CHECK(mat.emissive_tex != UINT32_MAX);

    remove_test_texture(tex_path);
}

TEST_CASE("Texture deduplication across material inputs") {
    auto tex_path = create_test_texture("pts_test_dedup.ppm");
    auto stage = create_textured_stage(tex_path);

    pts::rendering::RenderWorld world;
    auto scope = world.begin_sync();
    auto surface = pxr::UsdShadeShader::Get(stage, pxr::SdfPath("/Root/Mat/Surface"));
    auto mat = pts::rendering::read_preview_surface(surface, scope);

    // diffuse_tex, metallic_tex (ORM), roughness_tex (ORM), opacity_tex (ORM) all reference
    // the same file. ORM shares one texture, diffuse is a separate load of the same file.
    // Due to deduplication by path, they should share layer indices where paths match.
    // DiffuseTex and OrmTex both use the same file -> same layer index
    CHECK(mat.diffuse_tex == mat.metallic_tex);
    CHECK(mat.metallic_tex == mat.roughness_tex);
    CHECK(mat.roughness_tex == mat.opacity_tex);

    remove_test_texture(tex_path);
}

TEST_CASE("load_texture resolves filesystem paths via ArResolver") {
    auto tex_path = create_test_texture("pts_test_ar_resolve.ppm");

    pts::rendering::RenderWorld world;
    auto scope = world.begin_sync();

    // ArResolver handles filesystem paths -- exercises the stbi_load_from_memory path
    auto idx = scope.load_texture(tex_path);
    CHECK(idx != UINT32_MAX);
    CHECK(idx == 0);

    // Deduplication: same path returns same index
    CHECK(scope.load_texture(tex_path) == idx);

    // Nonexistent path returns UINT32_MAX
    CHECK(scope.load_texture("/nonexistent/image.png") == UINT32_MAX);

    remove_test_texture(tex_path);
}

TEST_CASE("load_texture handles EXR files") {
    auto exr_path = create_test_exr("pts_test_load.exr");

    pts::rendering::RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.load_texture(exr_path);
    CHECK(idx != UINT32_MAX);
    CHECK(idx == 0);

    // Deduplication: same path returns same index
    CHECK(scope.load_texture(exr_path) == idx);

    remove_test_texture(exr_path);
}

// --- GPU-dependent tests ---

#ifndef __EMSCRIPTEN__

TEST_CASE("Full pipeline: textured material through populate_from_stage") {
    auto tex_path = create_test_texture("pts_test_pipeline.ppm");
    auto stage = create_textured_stage(tex_path);

    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);

    REQUIRE(world.get_objects().size() == 1);
    REQUIRE(world.get_materials().size() == 1);

    auto& mat = world.get_materials()[0];
    CHECK(mat.diffuse_tex != UINT32_MAX);
    CHECK(mat.metallic_tex != UINT32_MAX);
    CHECK(mat.normal_tex != UINT32_MAX);
    CHECK(mat.emissive_tex != UINT32_MAX);
    CHECK(mat.ior == doctest::Approx(1.45f));

    remove_test_texture(tex_path);
}

#endif  // !__EMSCRIPTEN__

PTS_TEST_MAIN()
