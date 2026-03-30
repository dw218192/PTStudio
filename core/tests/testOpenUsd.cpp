#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <core/rendering/webgpu/device.h>
#include <embedded_test_resources.h>
#include <pxr/base/tf/notice.h>
#include <pxr/base/tf/weakBase.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/notice.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdUtils/usdzPackage.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "testApplication.h"

TEST_CASE("OpenUSD - Create in-memory stage") {
    auto stage = pxr::UsdStage::CreateInMemory();
    CHECK(stage);
}

TEST_CASE("OpenUSD - Load USDA from embedded string") {
    auto usda = test_resources::get_resource("scenes/test_cube.usda");
    REQUIRE(usda.has_value());

    auto layer = pxr::SdfLayer::CreateAnonymous(".usda");
    REQUIRE(layer);
    REQUIRE(layer->ImportFromString(std::string{usda.value()}));

    auto stage = pxr::UsdStage::Open(layer);
    REQUIRE(stage);

    SUBCASE("Default prim") {
        auto defaultPrim = stage->GetDefaultPrim();
        CHECK(defaultPrim.IsValid());
        CHECK(defaultPrim.GetName() == "Root");
    }

    SUBCASE("Prim hierarchy") {
        auto root = stage->GetPrimAtPath(pxr::SdfPath("/Root"));
        REQUIRE(root.IsValid());
        CHECK(bool(pxr::UsdGeomXform(root)));

        auto cube = stage->GetPrimAtPath(pxr::SdfPath("/Root/Cube"));
        REQUIRE(cube.IsValid());
        CHECK(bool(pxr::UsdGeomCube(cube)));
    }

    SUBCASE("Geometry attributes") {
        auto cube = pxr::UsdGeomCube::Get(stage, pxr::SdfPath("/Root/Cube"));
        REQUIRE(bool(cube));

        double size = 0;
        cube.GetSizeAttr().Get(&size);
        CHECK(size == doctest::Approx(2.0));
    }
}

namespace {
struct TestListener : pxr::TfWeakBase {
    bool got_resync{false};
    bool got_info_change{false};
    std::vector<std::string> resynced_paths;
    std::vector<std::string> changed_info_paths;

    void handle(const pxr::UsdNotice::ObjectsChanged& notice,
                const pxr::UsdStageWeakPtr& /*sender*/) {
        for (const auto& p : notice.GetResyncedPaths()) {
            got_resync = true;
            resynced_paths.push_back(p.GetString());
        }
        for (const auto& p : notice.GetChangedInfoOnlyPaths()) {
            got_info_change = true;
            changed_info_paths.push_back(p.GetString());
        }
    }
};
}  // namespace

// GPU-dependent tests — Device::create() requires native Dawn (not available in node.js)
#ifndef __EMSCRIPTEN__

TEST_CASE("populate_from_stage populates prim_path on ObjectData slots") {
    // Build a stage with a Mesh prim
    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    auto xform = pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/TestMesh"));

    // Minimal triangle: 3 points, 1 face with 3 vertices
    pxr::VtVec3fArray points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.GetPointsAttr().Set(points);
    pxr::VtIntArray face_counts = {3};
    mesh.GetFaceVertexCountsAttr().Set(face_counts);
    pxr::VtIntArray face_indices = {0, 1, 2};
    mesh.GetFaceVertexIndicesAttr().Set(face_indices);

    // Create device + populate
    auto logger = spdlog::stdout_color_mt("test_populate");
    auto device = pts::webgpu::Device::create(logger);
    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);
    world.upload_all_meshes(device);

    REQUIRE(world.get_objects().size() == 1);
    CHECK(world.get_objects()[0].get_prim_path() == pxr::SdfPath("/Root/TestMesh"));
    CHECK(world.get_meshes().size() == 1);
    CHECK(world.get_meshes()[0]->index_count == 3);

    spdlog::drop("test_populate");
}

TEST_CASE("USD ObjectsChanged fires on xform property edit") {
    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    auto xform = pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/TestMesh"));

    // Minimal geometry
    pxr::VtVec3fArray points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.GetPointsAttr().Set(points);
    pxr::VtIntArray face_counts = {3};
    mesh.GetFaceVertexCountsAttr().Set(face_counts);
    pxr::VtIntArray face_indices = {0, 1, 2};
    mesh.GetFaceVertexIndicesAttr().Set(face_indices);

    TestListener listener;
    auto key =
        pxr::TfNotice::Register(pxr::TfCreateWeakPtr(&listener), &TestListener::handle, stage);

    SUBCASE("xform property change fires info-only notice") {
        pxr::UsdGeomXformable xformable(mesh.GetPrim());
        xformable.ClearXformOpOrder();
        auto op = xformable.AddTransformOp();

        // Reset listener state after initial setup (Define calls fire resyncs)
        listener.got_resync = false;
        listener.got_info_change = false;
        listener.resynced_paths.clear();
        listener.changed_info_paths.clear();

        // Setting a value on an existing attribute fires an info-only change
        pxr::GfMatrix4d mat(1.0);
        mat[3][0] = 5.0;  // translate X
        op.Set(mat);

        CHECK(listener.got_info_change);
        CHECK(!listener.changed_info_paths.empty());
    }

    SUBCASE("defining a new prim fires resync notice") {
        listener.got_resync = false;
        listener.resynced_paths.clear();

        pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/NewMesh"));

        CHECK(listener.got_resync);
        bool found = false;
        for (const auto& p : listener.resynced_paths) {
            if (p.find("NewMesh") != std::string::npos) found = true;
        }
        CHECK(found);
    }

    pxr::TfNotice::Revoke(key);
}

TEST_CASE("Xform change updates ObjectData transform via notice pattern") {
    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    auto xform = pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/TestMesh"));

    pxr::VtVec3fArray points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.GetPointsAttr().Set(points);
    pxr::VtIntArray face_counts = {3};
    mesh.GetFaceVertexCountsAttr().Set(face_counts);
    pxr::VtIntArray face_indices = {0, 1, 2};
    mesh.GetFaceVertexIndicesAttr().Set(face_indices);

    auto logger = spdlog::stdout_color_mt("test_xform_change");
    auto device = pts::webgpu::Device::create(logger);
    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);
    world.upload_all_meshes(device);

    REQUIRE(world.get_objects().size() == 1);
    CHECK(world.get_objects()[0]->transform[0][3] == doctest::Approx(0.0f));

    // Simulate the full notice-driven update pattern used in EditorApplication:
    // 1. Notice fires with changed paths
    // 2. Categorize into resync vs xform-only
    // 3. Process dirty state: either full repopulate or per-prim transform update

    // Write a transform to the mesh prim (ClearXformOpOrder + AddTransformOp fires resync)
    pxr::UsdGeomXformable xformable(mesh.GetPrim());
    xformable.ClearXformOpOrder();
    xformable.AddTransformOp().Set(pxr::GfMatrix4d(1.0));

    // For the full resync path: clear + repopulate
    world.clear();
    pts::rendering::populate_from_stage(world, stage);
    world.upload_all_meshes(device);
    REQUIRE(world.get_objects().size() == 1);

    // Now test the xform-only fast path: Set() on existing op
    TestListener listener;
    auto key =
        pxr::TfNotice::Register(pxr::TfCreateWeakPtr(&listener), &TestListener::handle, stage);

    // Reuse existing op (mirrors optimized gizmo code)
    bool reset = false;
    auto ops = xformable.GetOrderedXformOps(&reset);
    REQUIRE(ops.size() == 1);
    pxr::GfMatrix4d mat(1.0);
    mat[3][0] = 7.0;  // translate X by 7
    ops[0].Set(mat);

    // Verify info-only notice fired (not resync)
    CHECK(listener.got_info_change);

    // Collect dirty prim paths from info-only changes
    std::vector<std::string> dirty_paths;
    for (const auto& p : listener.changed_info_paths) {
        auto sdf_path = pxr::SdfPath(p);
        auto prim_path = sdf_path.IsPropertyPath() ? sdf_path.GetPrimPath() : sdf_path;
        dirty_paths.push_back(prim_path.GetString());
    }

    // Process dirty paths via RenderWorld::update_transforms
    std::vector<pxr::SdfPath> sdf_dirty_paths;
    for (const auto& dp : dirty_paths) {
        sdf_dirty_paths.push_back(pxr::SdfPath(dp));
    }
    world.update_transforms(stage, sdf_dirty_paths);

    // Verify the transform was updated via the fast path.
    // GfMatrix4d[3][0] maps to glm[3][0] (direct copy, no transpose)
    CHECK(world.get_objects()[0]->transform[3][0] == doctest::Approx(7.0f));

    pxr::TfNotice::Revoke(key);
    spdlog::drop("test_xform_change");
}

TEST_CASE("Selection preserved across full resync by prim_path") {
    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
    auto mesh_a = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/MeshA"));
    auto mesh_b = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/MeshB"));

    // Minimal triangle geometry for both meshes
    pxr::VtVec3fArray points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    pxr::VtIntArray face_counts = {3};
    pxr::VtIntArray face_indices = {0, 1, 2};
    for (auto* m : {&mesh_a, &mesh_b}) {
        m->GetPointsAttr().Set(points);
        m->GetFaceVertexCountsAttr().Set(face_counts);
        m->GetFaceVertexIndicesAttr().Set(face_indices);
    }

    auto logger = spdlog::stdout_color_mt("test_selection_resync");
    auto device = pts::webgpu::Device::create(logger);
    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);
    world.upload_all_meshes(device);

    REQUIRE(world.get_objects().size() == 2);

    // Simulate selecting object at index 1 (MeshB)
    int selected_object = -1;
    for (int i = 0; i < static_cast<int>(world.get_objects().size()); ++i) {
        if (world.get_objects()[i].get_prim_path() == pxr::SdfPath("/Root/MeshB")) {
            selected_object = i;
            break;
        }
    }
    REQUIRE(selected_object >= 0);
    pxr::SdfPath selected_prim_path = world.get_objects()[selected_object].get_prim_path();

    // Simulate full resync (mirrors process_dirty_prims resync path)
    world.clear();
    pts::rendering::populate_from_stage(world, stage);
    world.upload_all_meshes(device);

    // Restore selection by prim_path
    int restored = -1;
    for (int i = 0; i < static_cast<int>(world.get_objects().size()); ++i) {
        if (world.get_objects()[i].get_prim_path() == selected_prim_path) {
            restored = i;
            break;
        }
    }

    CHECK(restored >= 0);
    CHECK(world.get_objects()[restored].get_prim_path() == pxr::SdfPath("/Root/MeshB"));

    spdlog::drop("test_selection_resync");
}

TEST_CASE("Selection lost when selected prim is removed during resync") {
    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/Mesh"));

    pxr::VtVec3fArray points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.GetPointsAttr().Set(points);
    pxr::VtIntArray face_counts = {3};
    mesh.GetFaceVertexCountsAttr().Set(face_counts);
    pxr::VtIntArray face_indices = {0, 1, 2};
    mesh.GetFaceVertexIndicesAttr().Set(face_indices);

    auto logger = spdlog::stdout_color_mt("test_selection_removed");
    auto device = pts::webgpu::Device::create(logger);
    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);
    world.upload_all_meshes(device);

    REQUIRE(world.get_objects().size() == 1);
    pxr::SdfPath selected_prim_path = world.get_objects()[0].get_prim_path();

    // Remove the prim from the stage
    stage->RemovePrim(pxr::SdfPath("/Root/Mesh"));

    // Full resync
    world.clear();
    pts::rendering::populate_from_stage(world, stage);
    world.upload_all_meshes(device);

    // Search for the removed prim
    int restored = -1;
    for (int i = 0; i < static_cast<int>(world.get_objects().size()); ++i) {
        if (world.get_objects()[i].get_prim_path() == selected_prim_path) {
            restored = i;
            break;
        }
    }

    CHECK(restored == -1);

    spdlog::drop("test_selection_removed");
}

TEST_CASE("Material extraction from UsdPreviewSurface") {
    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));

    // Create mesh with geometry
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/Mesh"));
    pxr::VtVec3fArray points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.GetPointsAttr().Set(points);
    mesh.GetFaceVertexCountsAttr().Set(pxr::VtIntArray{3});
    mesh.GetFaceVertexIndicesAttr().Set(pxr::VtIntArray{0, 1, 2});

    // Create material with UsdPreviewSurface shader
    auto material = pxr::UsdShadeMaterial::Define(stage, pxr::SdfPath("/Root/Mat"));
    auto shader = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Root/Mat/Surface"));
    shader.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdPreviewSurface")));
    shader.CreateInput(pxr::TfToken("diffuseColor"), pxr::SdfValueTypeNames->Color3f)
        .Set(pxr::GfVec3f(0.8f, 0.2f, 0.1f));
    shader.CreateInput(pxr::TfToken("metallic"), pxr::SdfValueTypeNames->Float).Set(0.9f);
    shader.CreateInput(pxr::TfToken("roughness"), pxr::SdfValueTypeNames->Float).Set(0.3f);
    shader.CreateInput(pxr::TfToken("opacity"), pxr::SdfValueTypeNames->Float).Set(0.7f);
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),
                                                   pxr::TfToken("surface"));

    // Bind material to mesh
    pxr::UsdShadeMaterialBindingAPI::Apply(mesh.GetPrim()).Bind(material);

    auto logger = spdlog::stdout_color_mt("test_material");
    auto device = pts::webgpu::Device::create(logger);
    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);
    world.upload_all_meshes(device);

    REQUIRE(world.get_objects().size() == 1);
    REQUIRE(world.get_materials().size() == 1);
    CHECK(world.get_objects()[0]->material_index == 1);

    auto& mat = world.get_materials()[0];
    CHECK(mat.diffuse_color.x == doctest::Approx(0.8f));
    CHECK(mat.diffuse_color.y == doctest::Approx(0.2f));
    CHECK(mat.diffuse_color.z == doctest::Approx(0.1f));
    CHECK(mat.metallic == doctest::Approx(0.9f));
    CHECK(mat.roughness == doctest::Approx(0.3f));
    CHECK(mat.opacity == doctest::Approx(0.7f));

    spdlog::drop("test_material");
}

TEST_CASE("Prim without material gets k_default_material") {
    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/Mesh"));
    pxr::VtVec3fArray points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.GetPointsAttr().Set(points);
    mesh.GetFaceVertexCountsAttr().Set(pxr::VtIntArray{3});
    mesh.GetFaceVertexIndicesAttr().Set(pxr::VtIntArray{0, 1, 2});

    auto logger = spdlog::stdout_color_mt("test_no_material");
    auto device = pts::webgpu::Device::create(logger);
    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);
    world.upload_all_meshes(device);

    REQUIRE(world.get_objects().size() == 1);
    CHECK(world.get_objects()[0]->material_index == pts::rendering::k_default_material);
    CHECK(world.get_materials().empty());

    spdlog::drop("test_no_material");
}

TEST_CASE("Shared material is deduplicated") {
    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));

    // Two meshes sharing one material
    auto mesh_a = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/MeshA"));
    auto mesh_b = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/MeshB"));
    pxr::VtVec3fArray points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    pxr::VtIntArray face_counts = {3};
    pxr::VtIntArray face_indices = {0, 1, 2};
    for (auto* m : {&mesh_a, &mesh_b}) {
        m->GetPointsAttr().Set(points);
        m->GetFaceVertexCountsAttr().Set(face_counts);
        m->GetFaceVertexIndicesAttr().Set(face_indices);
    }

    auto material = pxr::UsdShadeMaterial::Define(stage, pxr::SdfPath("/Root/SharedMat"));
    auto shader = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Root/SharedMat/Surface"));
    shader.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdPreviewSurface")));
    shader.CreateInput(pxr::TfToken("diffuseColor"), pxr::SdfValueTypeNames->Color3f)
        .Set(pxr::GfVec3f(0.5f, 0.5f, 0.5f));
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),
                                                   pxr::TfToken("surface"));

    pxr::UsdShadeMaterialBindingAPI::Apply(mesh_a.GetPrim()).Bind(material);
    pxr::UsdShadeMaterialBindingAPI::Apply(mesh_b.GetPrim()).Bind(material);

    auto logger = spdlog::stdout_color_mt("test_dedup_material");
    auto device = pts::webgpu::Device::create(logger);
    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);
    world.upload_all_meshes(device);

    REQUIRE(world.get_objects().size() == 2);
    CHECK(world.get_materials().size() == 1);
    CHECK(world.get_objects()[0]->material_index == world.get_objects()[1]->material_index);
    CHECK(world.get_objects()[0]->material_index == 1);

    spdlog::drop("test_dedup_material");
}

TEST_CASE("Default material is hidden from get_materials") {
    pts::rendering::RenderWorld world;
    CHECK(world.get_materials().empty());
    world.clear();
    CHECK(world.get_materials().empty());
}

TEST_CASE("Prim with displayColor creates material from displayColor") {
    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/Mesh"));
    pxr::VtVec3fArray points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.GetPointsAttr().Set(points);
    mesh.GetFaceVertexCountsAttr().Set(pxr::VtIntArray{3});
    mesh.GetFaceVertexIndicesAttr().Set(pxr::VtIntArray{0, 1, 2});

    // Set displayColor but no material binding
    auto primvars = pxr::UsdGeomPrimvarsAPI(mesh.GetPrim());
    auto color_pv =
        primvars.CreatePrimvar(pxr::TfToken("displayColor"), pxr::SdfValueTypeNames->Color3fArray);
    color_pv.Set(pxr::VtVec3fArray{{0.8f, 0.2f, 0.1f}});

    auto logger = spdlog::stdout_color_mt("test_display_color_mat");
    auto device = pts::webgpu::Device::create(logger);
    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);
    world.upload_all_meshes(device);

    REQUIRE(world.get_objects().size() == 1);
    REQUIRE(world.get_materials().size() == 1);
    auto mat_idx = world.get_objects()[0]->material_index;
    CHECK(mat_idx == 1);

    auto& mat = world.get_materials()[0];
    CHECK(mat.diffuse_color.x == doctest::Approx(0.8f));
    CHECK(mat.diffuse_color.y == doctest::Approx(0.2f));
    CHECK(mat.diffuse_color.z == doctest::Approx(0.1f));
    CHECK(mat.metallic == doctest::Approx(0.0f));
    CHECK(mat.roughness == doctest::Approx(0.5f));
    CHECK(mat.diffuse_tex == UINT32_MAX);

    spdlog::drop("test_display_color_mat");
}

TEST_CASE("Bound material takes precedence over displayColor") {
    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/Mesh"));
    pxr::VtVec3fArray points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.GetPointsAttr().Set(points);
    mesh.GetFaceVertexCountsAttr().Set(pxr::VtIntArray{3});
    mesh.GetFaceVertexIndicesAttr().Set(pxr::VtIntArray{0, 1, 2});

    // Set displayColor AND a bound material
    auto primvars = pxr::UsdGeomPrimvarsAPI(mesh.GetPrim());
    auto color_pv =
        primvars.CreatePrimvar(pxr::TfToken("displayColor"), pxr::SdfValueTypeNames->Color3fArray);
    color_pv.Set(pxr::VtVec3fArray{{1.0f, 0.0f, 0.0f}});

    auto material = pxr::UsdShadeMaterial::Define(stage, pxr::SdfPath("/Root/Mat"));
    auto shader = pxr::UsdShadeShader::Define(stage, pxr::SdfPath("/Root/Mat/Surface"));
    shader.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdPreviewSurface")));
    shader.CreateInput(pxr::TfToken("diffuseColor"), pxr::SdfValueTypeNames->Color3f)
        .Set(pxr::GfVec3f(0.0f, 0.5f, 1.0f));
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),
                                                   pxr::TfToken("surface"));
    pxr::UsdShadeMaterialBindingAPI::Apply(mesh.GetPrim()).Bind(material);

    auto logger = spdlog::stdout_color_mt("test_mat_over_display");
    auto device = pts::webgpu::Device::create(logger);
    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);
    world.upload_all_meshes(device);

    REQUIRE(world.get_objects().size() == 1);
    auto mat_idx = world.get_objects()[0]->material_index;
    REQUIRE(mat_idx > pts::rendering::k_default_material);
    REQUIRE(static_cast<std::size_t>(mat_idx - 1) < world.get_materials().size());
    // Bound material wins — displayColor is ignored
    auto& mat = world.get_materials()[mat_idx - 1];
    CHECK(mat.diffuse_color.x == doctest::Approx(0.0f));
    CHECK(mat.diffuse_color.y == doctest::Approx(0.5f));
    CHECK(mat.diffuse_color.z == doctest::Approx(1.0f));

    spdlog::drop("test_mat_over_display");
}

#endif  // !__EMSCRIPTEN__

TEST_CASE("Stage export-to-string round-trip preserves prims and transforms") {
    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Root/MyCube"));
    cube.GetSizeAttr().Set(3.0);

    // Set a transform on the cube
    pxr::UsdGeomXformable xformable(cube.GetPrim());
    pxr::GfMatrix4d mat(1.0);
    mat[3][0] = 10.0;  // translate X
    mat[3][1] = 20.0;  // translate Y
    mat[3][2] = 30.0;  // translate Z
    xformable.AddTransformOp().Set(mat);

    // Export to USDA string (same path as Save Scene on Emscripten)
    std::string usda;
    bool exported = stage->GetRootLayer()->ExportToString(&usda);
    REQUIRE(exported);
    REQUIRE(!usda.empty());

    // Re-import from the exported string
    auto layer2 = pxr::SdfLayer::CreateAnonymous(".usda");
    REQUIRE(layer2);
    REQUIRE(layer2->ImportFromString(usda));

    auto stage2 = pxr::UsdStage::Open(layer2);
    REQUIRE(stage2);

    // Verify prim exists with correct type
    auto cube2 = pxr::UsdGeomCube::Get(stage2, pxr::SdfPath("/Root/MyCube"));
    REQUIRE(bool(cube2));

    double size = 0;
    cube2.GetSizeAttr().Get(&size);
    CHECK(size == doctest::Approx(3.0));

    // Verify transform preserved
    pxr::UsdGeomXformable xformable2(cube2.GetPrim());
    pxr::GfMatrix4d reloaded;
    bool resetsXformStack = false;
    xformable2.GetLocalTransformation(&reloaded, &resetsXformStack, pxr::UsdTimeCode::Default());
    CHECK(reloaded[3][0] == doctest::Approx(10.0));
    CHECK(reloaded[3][1] == doctest::Approx(20.0));
    CHECK(reloaded[3][2] == doctest::Approx(30.0));
}

TEST_CASE("Eager xform normalization produces single TypeTransform op") {
    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/Mesh"));

    pxr::VtVec3fArray points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.GetPointsAttr().Set(points);
    pxr::VtIntArray face_counts = {3};
    mesh.GetFaceVertexCountsAttr().Set(face_counts);
    pxr::VtIntArray face_indices = {0, 1, 2};
    mesh.GetFaceVertexIndicesAttr().Set(face_indices);

    // Set up non-standard xform ops (translate + rotate instead of single transform)
    pxr::UsdGeomXformable xformable(mesh.GetPrim());
    xformable.AddTranslateOp().Set(pxr::GfVec3d(1.0, 2.0, 3.0));
    xformable.AddRotateXYZOp().Set(pxr::GfVec3f(0.0f, 45.0f, 0.0f));

    bool reset = false;
    auto ops_before = xformable.GetOrderedXformOps(&reset);
    CHECK(ops_before.size() == 2);

    // Normalize: mirrors EditorApplication::normalize_xform_ops
    // Must use GetLocalTransformation (not ComputeLocalToWorldTransform)
    // to avoid baking ancestor transforms into the local op.
    pxr::GfMatrix4d local_xf;
    bool resetsXformStack;
    xformable.GetLocalTransformation(&local_xf, &resetsXformStack, pxr::UsdTimeCode::Default());
    xformable.ClearXformOpOrder();
    xformable.AddTransformOp().Set(local_xf);

    auto ops_after = xformable.GetOrderedXformOps(&reset);
    REQUIRE(ops_after.size() == 1);
    CHECK(ops_after[0].GetOpType() == pxr::UsdGeomXformOp::TypeTransform);

    // Verify the local transform is preserved after normalization
    pxr::GfMatrix4d recomputed;
    xformable.GetLocalTransformation(&recomputed, &resetsXformStack, pxr::UsdTimeCode::Default());
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) CHECK(recomputed[r][c] == doctest::Approx(local_xf[r][c]));
}

TEST_CASE("Already-normalized xform ops are left unchanged") {
    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Root/Mesh"));

    pxr::UsdGeomXformable xformable(mesh.GetPrim());
    pxr::GfMatrix4d mat(1.0);
    mat[3][0] = 5.0;
    xformable.AddTransformOp().Set(mat);

    // Register a listener to detect any changes
    TestListener listener;
    auto key =
        pxr::TfNotice::Register(pxr::TfCreateWeakPtr(&listener), &TestListener::handle, stage);
    listener.got_resync = false;
    listener.got_info_change = false;

    // Check that normalization is a no-op when already normalized
    bool reset = false;
    auto ops = xformable.GetOrderedXformOps(&reset);
    bool already_normalized =
        (ops.size() == 1 && ops[0].GetOpType() == pxr::UsdGeomXformOp::TypeTransform);
    CHECK(already_normalized);

    // No ClearXformOpOrder + AddTransformOp should fire
    CHECK_FALSE(listener.got_resync);

    pxr::TfNotice::Revoke(key);
}

TEST_CASE("USDZ round-trip via UsdUtilsCreateNewUsdzPackage preserves geometry") {
    namespace fs = std::filesystem;

    auto stage = pxr::UsdStage::CreateInMemory();
    REQUIRE(stage);

    pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Root/MyCube"));
    cube.GetSizeAttr().Set(2.0);

    pxr::UsdGeomXformable xformable(cube.GetPrim());
    pxr::GfMatrix4d mat(1.0);
    mat[3][0] = 7.0;
    xformable.AddTransformOp().Set(mat);

    // Flatten and export to temp USDA
    auto tmp_dir = fs::temp_directory_path();
    auto tmp_usda = (tmp_dir / "_test_usdz.usda").string();
    auto tmp_usdz = (tmp_dir / "_test_usdz.usdz").string();

    auto flat = stage->Flatten();
    REQUIRE(flat);
    REQUIRE(flat->Export(tmp_usda));

    // Package as USDZ
    bool packaged = pxr::UsdUtilsCreateNewUsdzPackage(pxr::SdfAssetPath(tmp_usda), tmp_usdz);
    REQUIRE(packaged);

    // Verify the file is non-empty
    auto const nbytes = std::filesystem::file_size(tmp_usdz);
    CHECK(nbytes > 0);

    // Reopen the USDZ and verify geometry
    auto reopened = pxr::UsdStage::Open(tmp_usdz);
    REQUIRE(reopened);

    auto cube2 = pxr::UsdGeomCube::Get(reopened, pxr::SdfPath("/Root/MyCube"));
    REQUIRE(bool(cube2));

    double size = 0;
    cube2.GetSizeAttr().Get(&size);
    CHECK(size == doctest::Approx(2.0));

    pxr::UsdGeomXformable xformable2(cube2.GetPrim());
    pxr::GfMatrix4d reloaded;
    bool reset = false;
    xformable2.GetLocalTransformation(&reloaded, &reset, pxr::UsdTimeCode::Default());
    CHECK(reloaded[3][0] == doctest::Approx(7.0));

    // Cleanup
    fs::remove(tmp_usda);
    fs::remove(tmp_usdz);
}

PTS_TEST_MAIN()
