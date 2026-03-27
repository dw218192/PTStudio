#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/xform.h>

#include <string>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#ifndef DEMO_SCENES_DIR
#error "DEMO_SCENES_DIR must be defined — set by CMake to assets/scenes/"
#endif

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

static const std::string k_scenes_dir = STRINGIFY(DEMO_SCENES_DIR);

TEST_CASE("Demo USDZ - primitives.usdz opens and has a default prim") {
    auto stage = pxr::UsdStage::Open(k_scenes_dir + "/primitives.usdz");
    REQUIRE(stage);
    auto root = stage->GetDefaultPrim();
    CHECK(root.IsValid());
    CHECK(root.GetName() == "Root");
}

TEST_CASE("Demo USDZ - area_test.usdz opens and has a default prim") {
    auto stage = pxr::UsdStage::Open(k_scenes_dir + "/area_test.usdz");
    REQUIRE(stage);
    auto root = stage->GetDefaultPrim();
    CHECK(root.IsValid());
    CHECK(root.GetName() == "Root");
}

TEST_CASE("Demo USDZ - normal_map_test.usdz opens and has a default prim") {
    auto stage = pxr::UsdStage::Open(k_scenes_dir + "/normal_map_test.usdz");
    REQUIRE(stage);
    auto root = stage->GetDefaultPrim();
    CHECK(root.IsValid());
    CHECK(root.GetName() == "Root");
}
