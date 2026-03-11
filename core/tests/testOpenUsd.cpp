#include <embedded_test_resources.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/xform.h>

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

PTS_TEST_MAIN()
