#include <core/rendering/adapters/capsuleAdapter.h>
#include <core/rendering/adapters/coneAdapter.h>
#include <core/rendering/adapters/cubeAdapter.h>
#include <core/rendering/adapters/cylinderAdapter.h>
#include <core/rendering/adapters/registry.h>
#include <core/rendering/adapters/sphereAdapter.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <core/rendering/webgpu/device.h>
#include <embedded_test_resources.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cmath>
#include <string>

#include "testApplication.h"

// can_adapt tests — no GPU needed

TEST_CASE("Adapters do not cross-match prim types") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    auto sphere = pxr::UsdGeomSphere::Define(stage, pxr::SdfPath("/Sphere"));

    CHECK(!pts::rendering::SphereAdapter::instance().can_adapt(cube.GetPrim()));
    CHECK(!pts::rendering::CubeAdapter::instance().can_adapt(sphere.GetPrim()));
    CHECK(!pts::rendering::CylinderAdapter::instance().can_adapt(cube.GetPrim()));
    CHECK(!pts::rendering::ConeAdapter::instance().can_adapt(cube.GetPrim()));
    CHECK(!pts::rendering::CapsuleAdapter::instance().can_adapt(cube.GetPrim()));
}

// GPU-dependent tests — sync() uploads mesh data to the GPU

#ifndef __EMSCRIPTEN__

namespace {

struct TestFixture {
    std::shared_ptr<spdlog::logger> logger;
    pts::webgpu::Device device;
    pts::rendering::RenderWorld world;

    TestFixture(const char* name)
        : logger(spdlog::stdout_color_mt(name)), device(pts::webgpu::Device::create(logger)) {
    }

    ~TestFixture() {
        spdlog::drop(logger->name());
    }

    const pts::rendering::Mesh& synced_mesh() const {
        return world.meshes[world.objects[0].mesh_index];
    }
};

}  // namespace

TEST_CASE("CubeAdapter - basic cube") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(2.0);

    auto& adapter = pts::rendering::CubeAdapter::instance();
    CHECK(adapter.can_adapt(cube.GetPrim()));

    TestFixture f("test_cube_basic");
    adapter.sync(cube.GetPrim(), f.world, f.device);

    REQUIRE(f.world.objects.size() == 1);
    CHECK(f.world.objects[0].prim_path == "/Cube");
    // 36 indices (2 tris per face x 6 faces)
    CHECK(f.synced_mesh().index_count == 36);
    CHECK(f.synced_mesh().cpu_indices.size() == 36);
}

TEST_CASE("CubeAdapter - respects size attribute") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(4.0);

    TestFixture f("test_cube_size");
    pts::rendering::CubeAdapter::instance().sync(cube.GetPrim(), f.world, f.device);

    REQUIRE(f.world.objects.size() == 1);
    CHECK(f.synced_mesh().index_count == 36);
}

TEST_CASE("SphereAdapter - basic sphere") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto sphere = pxr::UsdGeomSphere::Define(stage, pxr::SdfPath("/Sphere"));
    sphere.GetRadiusAttr().Set(1.0);

    auto& adapter = pts::rendering::SphereAdapter::instance();
    CHECK(adapter.can_adapt(sphere.GetPrim()));

    TestFixture f("test_sphere_basic");
    adapter.sync(sphere.GetPrim(), f.world, f.device);

    REQUIRE(f.world.objects.size() == 1);
    CHECK(f.synced_mesh().index_count > 0);
}

TEST_CASE("CylinderAdapter - basic cylinder") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cyl = pxr::UsdGeomCylinder::Define(stage, pxr::SdfPath("/Cylinder"));

    auto& adapter = pts::rendering::CylinderAdapter::instance();
    CHECK(adapter.can_adapt(cyl.GetPrim()));

    TestFixture f("test_cylinder_basic");
    adapter.sync(cyl.GetPrim(), f.world, f.device);

    REQUIRE(f.world.objects.size() == 1);
    CHECK(f.synced_mesh().index_count > 0);
}

TEST_CASE("ConeAdapter - basic cone") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cone = pxr::UsdGeomCone::Define(stage, pxr::SdfPath("/Cone"));

    auto& adapter = pts::rendering::ConeAdapter::instance();
    CHECK(adapter.can_adapt(cone.GetPrim()));

    TestFixture f("test_cone_basic");
    adapter.sync(cone.GetPrim(), f.world, f.device);

    REQUIRE(f.world.objects.size() == 1);
    CHECK(f.synced_mesh().index_count > 0);
}

TEST_CASE("CapsuleAdapter - basic capsule") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cap = pxr::UsdGeomCapsule::Define(stage, pxr::SdfPath("/Capsule"));

    auto& adapter = pts::rendering::CapsuleAdapter::instance();
    CHECK(adapter.can_adapt(cap.GetPrim()));

    TestFixture f("test_capsule_basic");
    adapter.sync(cap.GetPrim(), f.world, f.device);

    REQUIRE(f.world.objects.size() == 1);
    CHECK(f.synced_mesh().index_count > 0);
}

TEST_CASE("test_cube.usda Cube prim is adapted by registry") {
    auto usda = test_resources::get_resource("scenes/test_cube.usda");
    REQUIRE(usda.has_value());

    auto layer = pxr::SdfLayer::CreateAnonymous(".usda");
    REQUIRE(layer);
    REQUIRE(layer->ImportFromString(std::string{usda.value()}));

    auto stage = pxr::UsdStage::Open(layer);
    REQUIRE(stage);

    auto cube_prim = stage->GetPrimAtPath(pxr::SdfPath("/Root/Cube"));
    REQUIRE(cube_prim.IsValid());

    TestFixture f("test_cube_registry");

    bool adapted = false;
    for (auto* adapter : pts::rendering::k_scene_adapters()) {
        if (!adapter->can_adapt(cube_prim)) continue;
        adapter->sync(cube_prim, f.world, f.device);
        REQUIRE(f.world.objects.size() == 1);
        CHECK(f.synced_mesh().index_count > 0);
        adapted = true;
        break;
    }
    CHECK(adapted);
}

TEST_CASE("sync_prim updates existing object") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(2.0);

    TestFixture f("test_sync_update");
    pts::rendering::populate_from_stage(f.world, stage, f.device);

    REQUIRE(f.world.objects.size() == 1);
    auto initial_version = f.world.mesh_version;

    // Re-sync the same prim — should update in place, not add a new object
    pts::rendering::sync_prim(f.world, stage, f.device, "/Cube");

    CHECK(f.world.objects.size() == 1);
    CHECK(f.world.mesh_version > initial_version);
}

TEST_CASE("remove_prim frees object and mesh slots") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(2.0);

    TestFixture f("test_remove_prim");
    pts::rendering::populate_from_stage(f.world, stage, f.device);

    REQUIRE(f.world.objects.size() == 1);
    CHECK(f.world.objects[0].active);
    auto initial_version = f.world.mesh_version;

    pts::rendering::remove_prim(f.world, "/Cube");

    CHECK(!f.world.objects[0].active);
    CHECK(f.world.find_object_by_prim("/Cube") == -1);
    CHECK(f.world.mesh_version > initial_version);
}

TEST_CASE("sync_prim with invalid path calls remove_prim") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(2.0);

    TestFixture f("test_sync_invalid");
    pts::rendering::populate_from_stage(f.world, stage, f.device);

    REQUIRE(f.world.objects.size() == 1);

    // Remove from stage, then sync — should remove from world
    stage->RemovePrim(pxr::SdfPath("/Cube"));
    pts::rendering::sync_prim(f.world, stage, f.device, "/Cube");

    CHECK(!f.world.objects[0].active);
    CHECK(f.world.find_object_by_prim("/Cube") == -1);
}

#endif  // !__EMSCRIPTEN__

PTS_TEST_MAIN()
