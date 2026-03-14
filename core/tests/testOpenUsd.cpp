#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <core/rendering/webgpu/device.h>
#include <embedded_test_resources.h>
#include <pxr/base/tf/notice.h>
#include <pxr/base/tf/weakBase.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/notice.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <spdlog/sinks/stdout_color_sinks.h>

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

TEST_CASE("populate_from_stage populates prim_path on RenderObjects") {
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
    pts::rendering::populate_from_stage(world, stage, device);

    REQUIRE(world.objects.size() == 1);
    CHECK(world.objects[0].prim_path == "/Root/TestMesh");
    CHECK(world.meshes.size() == 1);
    CHECK(world.meshes[0].index_count == 3);

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

TEST_CASE("Xform change updates RenderObject transform via notice pattern") {
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
    pts::rendering::populate_from_stage(world, stage, device);

    REQUIRE(world.objects.size() == 1);
    CHECK(world.objects[0].transform[0][3] == doctest::Approx(0.0f));

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
    pts::rendering::populate_from_stage(world, stage, device);
    REQUIRE(world.objects.size() == 1);

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

    // Process dirty paths: update RenderObject transforms from USD
    for (const auto& dp : dirty_paths) {
        for (auto& obj : world.objects) {
            if (obj.prim_path != dp) continue;
            auto prim = stage->GetPrimAtPath(pxr::SdfPath(obj.prim_path));
            REQUIRE(prim.IsValid());
            pxr::UsdGeomXformable xf(prim);
            REQUIRE(bool(xf));
            auto gf = xf.ComputeLocalToWorldTransform(pxr::UsdTimeCode::Default());
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c) obj.transform[c][r] = static_cast<float>(gf[r][c]);
        }
    }

    // Verify the transform was updated via the fast path.
    // GfMatrix4d[3][0] (translation X, row-major) maps to glm[0][3] (col 0, row 3)
    // via the copy formula: transform[col][row] = xf[row][col]
    CHECK(world.objects[0].transform[0][3] == doctest::Approx(7.0f));

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
    pts::rendering::populate_from_stage(world, stage, device);

    REQUIRE(world.objects.size() == 2);

    // Simulate selecting object at index 1 (MeshB)
    int selected_object = -1;
    for (int i = 0; i < static_cast<int>(world.objects.size()); ++i) {
        if (world.objects[i].prim_path == "/Root/MeshB") {
            selected_object = i;
            break;
        }
    }
    REQUIRE(selected_object >= 0);
    std::string selected_prim_path = world.objects[selected_object].prim_path;

    // Simulate full resync (mirrors process_dirty_prims resync path)
    world.clear();
    pts::rendering::populate_from_stage(world, stage, device);

    // Restore selection by prim_path
    int restored = -1;
    for (int i = 0; i < static_cast<int>(world.objects.size()); ++i) {
        if (world.objects[i].prim_path == selected_prim_path) {
            restored = i;
            break;
        }
    }

    CHECK(restored >= 0);
    CHECK(world.objects[restored].prim_path == "/Root/MeshB");

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
    pts::rendering::populate_from_stage(world, stage, device);

    REQUIRE(world.objects.size() == 1);
    std::string selected_prim_path = world.objects[0].prim_path;

    // Remove the prim from the stage
    stage->RemovePrim(pxr::SdfPath("/Root/Mesh"));

    // Full resync
    world.clear();
    pts::rendering::populate_from_stage(world, stage, device);

    // Search for the removed prim
    int restored = -1;
    for (int i = 0; i < static_cast<int>(world.objects.size()); ++i) {
        if (world.objects[i].prim_path == selected_prim_path) {
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
    pts::rendering::populate_from_stage(world, stage, device);

    REQUIRE(world.objects.size() == 1);
    REQUIRE(world.materials.size() == 1);
    CHECK(world.objects[0].material_index == 0);

    auto& mat = world.materials[0];
    CHECK(mat.diffuse_color.x == doctest::Approx(0.8f));
    CHECK(mat.diffuse_color.y == doctest::Approx(0.2f));
    CHECK(mat.diffuse_color.z == doctest::Approx(0.1f));
    CHECK(mat.metallic == doctest::Approx(0.9f));
    CHECK(mat.roughness == doctest::Approx(0.3f));
    CHECK(mat.opacity == doctest::Approx(0.7f));

    spdlog::drop("test_material");
}

TEST_CASE("Prim without material gets k_no_material") {
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
    pts::rendering::populate_from_stage(world, stage, device);

    REQUIRE(world.objects.size() == 1);
    CHECK(world.objects[0].material_index == pts::rendering::k_no_material);
    CHECK(world.materials.empty());

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
    pts::rendering::populate_from_stage(world, stage, device);

    REQUIRE(world.objects.size() == 2);
    CHECK(world.materials.size() == 1);
    CHECK(world.objects[0].material_index == world.objects[1].material_index);
    CHECK(world.objects[0].material_index == 0);

    spdlog::drop("test_dedup_material");
}

#endif  // !__EMSCRIPTEN__

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
    auto computed = xformable.ComputeLocalToWorldTransform(pxr::UsdTimeCode::Default());
    xformable.ClearXformOpOrder();
    xformable.AddTransformOp().Set(computed);

    auto ops_after = xformable.GetOrderedXformOps(&reset);
    REQUIRE(ops_after.size() == 1);
    CHECK(ops_after[0].GetOpType() == pxr::UsdGeomXformOp::TypeTransform);

    // Verify the computed transform is preserved
    auto recomputed = xformable.ComputeLocalToWorldTransform(pxr::UsdTimeCode::Default());
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) CHECK(recomputed[r][c] == doctest::Approx(computed[r][c]));
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

PTS_TEST_MAIN()
