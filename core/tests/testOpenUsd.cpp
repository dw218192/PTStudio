#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <core/rendering/webgpu/device.h>
#include <embedded_test_resources.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <spdlog/sinks/stdout_color_sinks.h>

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

PTS_TEST_MAIN()
