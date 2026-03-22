#include <core/backgroundTask.h>
#include <core/rendering/adapters/capsuleAdapter.h>
#include <core/rendering/adapters/coneAdapter.h>
#include <core/rendering/adapters/cubeAdapter.h>
#include <core/rendering/adapters/cylinderAdapter.h>
#include <core/rendering/adapters/lightAdapter.h>
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

TEST_CASE("populate_from_stage with progress builds RenderWorld") {
    auto stage = pxr::UsdStage::CreateInMemory();
    pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/CubeA"));
    pxr::UsdGeomSphere::Define(stage, pxr::SdfPath("/SphereB"));

    pts::TaskProgress progress;
    auto world = pts::rendering::populate_from_stage(stage, progress);

    // Progress should reach 1.0
    CHECK(progress.progress() == doctest::Approx(1.0f));
    // Status should be the last prim path processed
    CHECK(!progress.status().empty());

    // Both prims should be synced (pseudoroot is not adapted, but the two shapes are)
    auto objects = world.get_objects();
    size_t active_count = 0;
    for (const auto& obj : objects) {
        if (obj.active()) ++active_count;
    }
    CHECK(active_count == 2);

    // CPU data present, no GPU buffers
    auto meshes = world.get_meshes();
    for (const auto& obj : objects) {
        if (!obj.active()) continue;
        const auto& mesh = meshes[obj->mesh_index];
        CHECK(mesh->cpu_vertices.size() > 0);
        CHECK(mesh->cpu_indices.size() > 0);
        CHECK(mesh->vertex_buffer.handle() == nullptr);
        CHECK(mesh->index_buffer.handle() == nullptr);
    }
}

// PrimFactory tests — no GPU needed

TEST_CASE("Geometry adapters each return exactly one factory") {
    CHECK(pts::rendering::CubeAdapter::instance().get_factories().size() == 1);
    CHECK(pts::rendering::SphereAdapter::instance().get_factories().size() == 1);
    CHECK(pts::rendering::CylinderAdapter::instance().get_factories().size() == 1);
    CHECK(pts::rendering::ConeAdapter::instance().get_factories().size() == 1);
    CHECK(pts::rendering::CapsuleAdapter::instance().get_factories().size() == 1);
}

TEST_CASE("LightAdapter returns 5 factories") {
    auto factories = pts::rendering::LightAdapter::instance().get_factories();
    REQUIRE(factories.size() == 5);
    CHECK(factories[0].category == "Lights");
    CHECK(factories[0].display_name == "Distant Light");
    CHECK(factories[1].display_name == "Sphere Light");
    CHECK(factories[2].display_name == "Rect Light");
    CHECK(factories[3].display_name == "Disk Light");
    CHECK(factories[4].display_name == "Dome Light");
}

TEST_CASE("Geometry factory categories and base names are correct") {
    auto cube_fac = pts::rendering::CubeAdapter::instance().get_factories()[0];
    CHECK(cube_fac.category == "Geometry");
    CHECK(cube_fac.display_name == "Cube");
    CHECK(cube_fac.base_name == "Cube");

    auto sphere_fac = pts::rendering::SphereAdapter::instance().get_factories()[0];
    CHECK(sphere_fac.category == "Geometry");
    CHECK(sphere_fac.base_name == "Sphere");
}

TEST_CASE("Factory define function creates valid prims") {
    auto stage = pxr::UsdStage::CreateInMemory();

    auto cube_fac = pts::rendering::CubeAdapter::instance().get_factories()[0];
    auto prim = cube_fac.define(stage, pxr::SdfPath("/TestCube"));
    REQUIRE(prim.IsValid());
    CHECK(prim.IsA<pxr::UsdGeomCube>());
    CHECK(pts::rendering::CubeAdapter::instance().can_adapt(prim));

    auto sphere_fac = pts::rendering::SphereAdapter::instance().get_factories()[0];
    auto sphere_prim = sphere_fac.define(stage, pxr::SdfPath("/TestSphere"));
    REQUIRE(sphere_prim.IsValid());
    CHECK(sphere_prim.IsA<pxr::UsdGeomSphere>());
}

TEST_CASE("Light factory define functions create valid light prims") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto factories = pts::rendering::LightAdapter::instance().get_factories();

    for (const auto& factory : factories) {
        auto path = pxr::SdfPath("/" + factory.base_name);
        auto prim = factory.define(stage, path);
        REQUIRE(prim.IsValid());
        CHECK(pts::rendering::LightAdapter::instance().can_adapt(prim));
    }
}

TEST_CASE("Registry collects all factories from adapters") {
    std::vector<pts::rendering::PrimFactory> all;
    for (auto* adapter : pts::rendering::k_scene_adapters()) {
        auto factories = adapter->get_factories();
        all.insert(all.end(), factories.begin(), factories.end());
    }
    // 5 geometry + 5 lights = 10
    CHECK(all.size() == 10);
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

    const pts::rendering::MeshData& synced_mesh() const {
        auto objects = world.get_objects();
        auto meshes = world.get_meshes();
        return meshes[objects[0]->mesh_index].data();
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
    {
        auto scope = f.world.begin_sync();
        adapter.sync(cube.GetPrim(), scope);
    }
    f.world.upload_all_meshes(f.device);

    REQUIRE(f.world.get_objects().size() == 1);
    CHECK(f.world.get_objects()[0].get_prim_path() == "/Cube");
    // 36 indices (2 tris per face x 6 faces)
    CHECK(f.synced_mesh().index_count == 36);
    CHECK(f.synced_mesh().cpu_indices.size() == 36);
}

TEST_CASE("CubeAdapter - respects size attribute") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(4.0);

    TestFixture f("test_cube_size");
    {
        auto scope = f.world.begin_sync();
        pts::rendering::CubeAdapter::instance().sync(cube.GetPrim(), scope);
    }
    f.world.upload_all_meshes(f.device);

    REQUIRE(f.world.get_objects().size() == 1);
    CHECK(f.synced_mesh().index_count == 36);
}

TEST_CASE("SphereAdapter - basic sphere") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto sphere = pxr::UsdGeomSphere::Define(stage, pxr::SdfPath("/Sphere"));
    sphere.GetRadiusAttr().Set(1.0);

    auto& adapter = pts::rendering::SphereAdapter::instance();
    CHECK(adapter.can_adapt(sphere.GetPrim()));

    TestFixture f("test_sphere_basic");
    {
        auto scope = f.world.begin_sync();
        adapter.sync(sphere.GetPrim(), scope);
    }
    f.world.upload_all_meshes(f.device);

    REQUIRE(f.world.get_objects().size() == 1);
    CHECK(f.synced_mesh().index_count > 0);
}

TEST_CASE("CylinderAdapter - basic cylinder") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cyl = pxr::UsdGeomCylinder::Define(stage, pxr::SdfPath("/Cylinder"));

    auto& adapter = pts::rendering::CylinderAdapter::instance();
    CHECK(adapter.can_adapt(cyl.GetPrim()));

    TestFixture f("test_cylinder_basic");
    {
        auto scope = f.world.begin_sync();
        adapter.sync(cyl.GetPrim(), scope);
    }
    f.world.upload_all_meshes(f.device);

    REQUIRE(f.world.get_objects().size() == 1);
    CHECK(f.synced_mesh().index_count > 0);
}

TEST_CASE("ConeAdapter - basic cone") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cone = pxr::UsdGeomCone::Define(stage, pxr::SdfPath("/Cone"));

    auto& adapter = pts::rendering::ConeAdapter::instance();
    CHECK(adapter.can_adapt(cone.GetPrim()));

    TestFixture f("test_cone_basic");
    {
        auto scope = f.world.begin_sync();
        adapter.sync(cone.GetPrim(), scope);
    }
    f.world.upload_all_meshes(f.device);

    REQUIRE(f.world.get_objects().size() == 1);
    CHECK(f.synced_mesh().index_count > 0);
}

TEST_CASE("CapsuleAdapter - basic capsule") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cap = pxr::UsdGeomCapsule::Define(stage, pxr::SdfPath("/Capsule"));

    auto& adapter = pts::rendering::CapsuleAdapter::instance();
    CHECK(adapter.can_adapt(cap.GetPrim()));

    TestFixture f("test_capsule_basic");
    {
        auto scope = f.world.begin_sync();
        adapter.sync(cap.GetPrim(), scope);
    }
    f.world.upload_all_meshes(f.device);

    REQUIRE(f.world.get_objects().size() == 1);
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
    auto scope = f.world.begin_sync();

    bool adapted = false;
    for (auto* adapter : pts::rendering::k_scene_adapters()) {
        if (!adapter->can_adapt(cube_prim)) continue;
        adapter->sync(cube_prim, scope);
        REQUIRE(f.world.get_objects().size() == 1);
        CHECK(f.synced_mesh().index_count > 0);
        adapted = true;
        break;
    }
    CHECK(adapted);
}

TEST_CASE("CPU-only sync populates vertices and indices without GPU buffers") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(2.0);

    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);

    REQUIRE(world.get_objects().size() == 1);
    CHECK(world.get_objects()[0].get_prim_path() == "/Cube");

    auto const& mesh = world.get_meshes()[world.get_objects()[0]->mesh_index].data();
    CHECK(mesh.index_count == 36);
    CHECK(mesh.cpu_indices.size() == 36);
    CHECK(mesh.cpu_vertices.size() > 0);
    // GPU buffers should not be created
    CHECK(mesh.vertex_buffer.handle() == nullptr);
    CHECK(mesh.index_buffer.handle() == nullptr);
}

TEST_CASE("sync_prim updates existing object") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(2.0);

    TestFixture f("test_sync_update");
    pts::rendering::populate_from_stage(f.world, stage);
    f.world.upload_all_meshes(f.device);

    REQUIRE(f.world.get_objects().size() == 1);
    auto initial_version = f.world.get_mesh_version();

    // Re-sync the same prim — should update in place, not add a new object
    {
        auto scope = f.world.begin_sync();
        pts::rendering::sync_prim(scope, stage, pxr::SdfPath("/Cube"));
    }

    CHECK(f.world.get_objects().size() == 1);
    CHECK(f.world.get_mesh_version() > initial_version);
}

TEST_CASE("remove_prim frees object and mesh slots") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(2.0);

    TestFixture f("test_remove_prim");
    pts::rendering::populate_from_stage(f.world, stage);
    f.world.upload_all_meshes(f.device);

    REQUIRE(f.world.get_objects().size() == 1);
    CHECK(f.world.get_objects()[0].active());
    auto initial_version = f.world.get_mesh_version();

    {
        auto scope = f.world.begin_sync();
        pts::rendering::remove_prim(scope, pxr::SdfPath("/Cube"));
    }

    CHECK(!f.world.get_objects()[0].active());
    CHECK(f.world.find_object_by_prim("/Cube") == -1);
    CHECK(f.world.get_mesh_version() > initial_version);
}

TEST_CASE("sync_prim with invalid path calls remove_prim") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(2.0);

    TestFixture f("test_sync_invalid");
    pts::rendering::populate_from_stage(f.world, stage);
    f.world.upload_all_meshes(f.device);

    REQUIRE(f.world.get_objects().size() == 1);

    // Remove from stage, then sync — should remove from world
    stage->RemovePrim(pxr::SdfPath("/Cube"));
    {
        auto scope = f.world.begin_sync();
        pts::rendering::sync_prim(scope, stage, pxr::SdfPath("/Cube"));
    }

    CHECK(!f.world.get_objects()[0].active());
    CHECK(f.world.find_object_by_prim("/Cube") == -1);
}

#endif  // !__EMSCRIPTEN__

PTS_TEST_MAIN()
