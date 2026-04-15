#include <core/rendering/adapters/capsuleAdapter.h>
#include <core/rendering/adapters/coneAdapter.h>
#include <core/rendering/adapters/cubeAdapter.h>
#include <core/rendering/adapters/cylinderAdapter.h>
#include <core/rendering/adapters/lightAdapter.h>
#include <core/rendering/adapters/meshAdapter.h>
#include <core/rendering/adapters/registry.h>
#include <core/rendering/adapters/sphereAdapter.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <core/rendering/webgpu/device.h>
#include <core/worker.h>
#include <embedded_test_resources.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cmath>
#include <string>

#include "testApplication.h"

// can_adapt tests -- no GPU needed

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

TEST_CASE("MeshAdapter can_adapt recognizes GeomSubset prims") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Mesh"));
    auto subset = pxr::UsdGeomSubset::Define(stage, pxr::SdfPath("/Mesh/Subset"));
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));

    auto& adapter = pts::rendering::MeshAdapter::instance();
    CHECK(adapter.can_adapt(mesh.GetPrim()));
    CHECK(adapter.can_adapt(subset.GetPrim()));
    CHECK(!adapter.can_adapt(cube.GetPrim()));
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
    const auto& objects = world.get_objects();
    size_t active_count = 0;
    for (const auto& e : objects.span_raw()) {
        if (e.active) ++active_count;
    }
    CHECK(active_count == 2);

    // CPU data present, no GPU buffers
    const auto& meshes = world.get_meshes();
    for (const auto& e : objects.span_raw()) {
        if (!e.active) continue;
        const auto& mesh = meshes.at(e.value.mesh_index);
        CHECK(mesh.cpu_vertices.size() > 0);
        CHECK(mesh.cpu_indices.size() > 0);
        CHECK(mesh.vertex_buffer.handle() == nullptr);
        CHECK(mesh.index_buffer.handle() == nullptr);
    }
}

// PrimFactory tests -- no GPU needed

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

TEST_CASE("Registry collects factories from adapters") {
    std::vector<pts::rendering::PrimFactory> all;
    for (auto* adapter : pts::rendering::k_scene_adapters()) {
        auto factories = adapter->get_factories();
        all.insert(all.end(), factories.begin(), factories.end());
    }
    // Every adapter contributes at least one factory
    CHECK(all.size() >= pts::rendering::k_scene_adapters().size());

    // Verify expected categories are present
    bool has_geometry = false;
    bool has_lights = false;
    for (const auto& f : all) {
        if (f.category == "Geometry") has_geometry = true;
        if (f.category == "Lights") has_lights = true;
    }
    CHECK(has_geometry);
    CHECK(has_lights);
}

// GPU-dependent tests -- sync() uploads mesh data to the GPU

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
        auto& objects = world.get_objects();
        auto& meshes = world.get_meshes();
        return meshes.at(objects.at(0).mesh_index);
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
    CHECK(f.world.find_object_by_prim(pxr::SdfPath("/Cube")) >= 0);
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
        CHECK(f.world.get_objects().size() == 1);
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
    CHECK(world.find_object_by_prim(pxr::SdfPath("/Cube")) >= 0);

    auto const& obj = world.get_objects().at(0);
    auto const& mesh = world.get_meshes().at(obj.mesh_index);
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

    // Re-sync the same prim -- should update in place, not add a new object
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

    REQUIRE(f.world.get_objects().capacity() >= 1);
    CHECK(f.world.get_objects().active_at(0));
    auto initial_version = f.world.get_mesh_version();

    {
        auto scope = f.world.begin_sync();
        pts::rendering::remove_prim(scope, pxr::SdfPath("/Cube"));
    }

    CHECK(!f.world.get_objects().active_at(0));
    CHECK(f.world.find_object_by_prim(pxr::SdfPath("/Cube")) == -1);
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

    // Remove from stage, then sync -- should remove from world
    stage->RemovePrim(pxr::SdfPath("/Cube"));
    {
        auto scope = f.world.begin_sync();
        pts::rendering::sync_prim(scope, stage, pxr::SdfPath("/Cube"));
    }

    CHECK(!f.world.get_objects().active_at(0));
    CHECK(f.world.find_object_by_prim(pxr::SdfPath("/Cube")) == -1);
}

TEST_CASE("sync_object reads UsdGeomImageable visibility") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto visible_cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Visible"));
    auto hidden_cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Hidden"));
    pxr::UsdGeomImageable(hidden_cube).GetVisibilityAttr().Set(pxr::UsdGeomTokens->invisible);

    TestFixture f("test_visibility");
    pts::rendering::populate_from_stage(f.world, stage);

    const auto& objects = f.world.get_objects();
    REQUIRE(objects.size() == 2);

    int vis_idx = f.world.find_object_by_prim(pxr::SdfPath("/Visible"));
    int hid_idx = f.world.find_object_by_prim(pxr::SdfPath("/Hidden"));
    REQUIRE(vis_idx >= 0);
    REQUIRE(hid_idx >= 0);

    CHECK(objects.at(static_cast<uint32_t>(vis_idx)).visible == true);
    CHECK(objects.at(static_cast<uint32_t>(hid_idx)).visible == false);
}

TEST_CASE("visibility updates on re-sync") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));

    TestFixture f("test_visibility_resync");
    pts::rendering::populate_from_stage(f.world, stage);

    REQUIRE(f.world.get_objects().size() == 1);
    CHECK(f.world.get_objects().at(0).visible == true);

    // Hide the cube and re-sync
    pxr::UsdGeomImageable(cube).GetVisibilityAttr().Set(pxr::UsdGeomTokens->invisible);
    {
        auto scope = f.world.begin_sync();
        pts::rendering::sync_prim(scope, stage, pxr::SdfPath("/Cube"));
    }

    CHECK(f.world.get_objects().at(0).visible == false);

    // Make visible again via "inherited"
    pxr::UsdGeomImageable(cube).GetVisibilityAttr().Set(pxr::UsdGeomTokens->inherited);
    {
        auto scope = f.world.begin_sync();
        pts::rendering::sync_prim(scope, stage, pxr::SdfPath("/Cube"));
    }

    CHECK(f.world.get_objects().at(0).visible == true);
}

// --- GeomSubset material binding tests ---

namespace {

/// Helper: create a quad mesh with 4 triangular faces at /Mesh.
/// Layout: 4 triangles sharing a center vertex (fan around {0,0,0}).
///   face 0: v0-v1-v4, face 1: v1-v2-v4, face 2: v2-v3-v4, face 3: v3-v0-v4
pxr::UsdGeomMesh define_quad_fan_mesh(const pxr::UsdStageRefPtr& stage) {
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Mesh"));
    mesh.GetPointsAttr().Set(
        pxr::VtVec3fArray{{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, {0, 0, 0}});
    mesh.GetFaceVertexCountsAttr().Set(pxr::VtIntArray{3, 3, 3, 3});
    mesh.GetFaceVertexIndicesAttr().Set(pxr::VtIntArray{0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4});
    return mesh;
}

size_t count_active_objects(const pts::rendering::RenderWorld& world) {
    return world.get_objects().size();
}

}  // namespace

TEST_CASE("GeomSubset materialBind creates per-subset objects") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto mesh = define_quad_fan_mesh(stage);

    // Two subsets, each covering 2 faces -- full coverage, no remainder.
    auto sub_a = pxr::UsdGeomSubset::Define(stage, pxr::SdfPath("/Mesh/SubA"));
    sub_a.GetFamilyNameAttr().Set(pxr::TfToken("materialBind"));
    sub_a.GetElementTypeAttr().Set(pxr::TfToken("face"));
    sub_a.GetIndicesAttr().Set(pxr::VtIntArray{0, 1});

    auto sub_b = pxr::UsdGeomSubset::Define(stage, pxr::SdfPath("/Mesh/SubB"));
    sub_b.GetFamilyNameAttr().Set(pxr::TfToken("materialBind"));
    sub_b.GetElementTypeAttr().Set(pxr::TfToken("face"));
    sub_b.GetIndicesAttr().Set(pxr::VtIntArray{2, 3});

    // Bind distinct materials to each subset.
    auto mat_a = pxr::UsdShadeMaterial::Define(stage, pxr::SdfPath("/MatA"));
    auto mat_b = pxr::UsdShadeMaterial::Define(stage, pxr::SdfPath("/MatB"));
    pxr::UsdShadeMaterialBindingAPI::Apply(sub_a.GetPrim()).Bind(mat_a);
    pxr::UsdShadeMaterialBindingAPI::Apply(sub_b.GetPrim()).Bind(mat_b);

    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);

    CHECK(count_active_objects(world) == 2);
    CHECK(world.find_object_by_prim(pxr::SdfPath("/Mesh/SubA")) >= 0);
    CHECK(world.find_object_by_prim(pxr::SdfPath("/Mesh/SubB")) >= 0);

    // Each subset has 2 faces -> 2 triangles -> 6 indices.
    const auto& meshes = world.get_meshes();
    const auto& objects = world.get_objects();
    objects.for_each([&](const pxr::SdfPath&, const pts::rendering::ObjectData& obj) {
        CHECK(meshes.at(obj.mesh_index).index_count == 6);
    });

    // Materials should be distinct.
    int ia = world.find_object_by_prim(pxr::SdfPath("/Mesh/SubA"));
    int ib = world.find_object_by_prim(pxr::SdfPath("/Mesh/SubB"));
    CHECK(objects.at(static_cast<uint32_t>(ia)).material_index !=
          objects.at(static_cast<uint32_t>(ib)).material_index);
}

TEST_CASE("GeomSubset with remainder emits mesh-level object for uncovered faces") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto mesh = define_quad_fan_mesh(stage);

    // One subset covering faces 0,1 -- faces 2,3 are remainder.
    auto sub = pxr::UsdGeomSubset::Define(stage, pxr::SdfPath("/Mesh/Sub"));
    sub.GetFamilyNameAttr().Set(pxr::TfToken("materialBind"));
    sub.GetElementTypeAttr().Set(pxr::TfToken("face"));
    sub.GetIndicesAttr().Set(pxr::VtIntArray{0, 1});

    auto mat = pxr::UsdShadeMaterial::Define(stage, pxr::SdfPath("/Mat"));
    pxr::UsdShadeMaterialBindingAPI::Apply(sub.GetPrim()).Bind(mat);

    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);

    // 1 subset object + 1 remainder object.
    CHECK(count_active_objects(world) == 2);
    CHECK(world.find_object_by_prim(pxr::SdfPath("/Mesh/Sub")) >= 0);
    CHECK(world.find_object_by_prim(pxr::SdfPath("/Mesh")) >= 0);

    const auto& meshes = world.get_meshes();
    const auto& objects = world.get_objects();

    int sub_idx = world.find_object_by_prim(pxr::SdfPath("/Mesh/Sub"));
    int rem_idx = world.find_object_by_prim(pxr::SdfPath("/Mesh"));
    CHECK(meshes.at(objects.at(static_cast<uint32_t>(sub_idx)).mesh_index).index_count == 6);
    CHECK(meshes.at(objects.at(static_cast<uint32_t>(rem_idx)).mesh_index).index_count == 6);
}

TEST_CASE("Mesh without GeomSubsets creates single object (no regression)") {
    auto stage = pxr::UsdStage::CreateInMemory();
    define_quad_fan_mesh(stage);

    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);

    CHECK(count_active_objects(world) == 1);
    CHECK(world.find_object_by_prim(pxr::SdfPath("/Mesh")) >= 0);

    const auto& objects = world.get_objects();
    const auto& meshes = world.get_meshes();
    // 4 faces x 1 tri each = 4 triangles = 12 indices.
    CHECK(meshes.at(objects.at(0).mesh_index).index_count == 12);
}

TEST_CASE("Non-materialBind subsets are ignored (mesh treated as whole)") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto mesh = define_quad_fan_mesh(stage);

    // Subset with a different family -- should be ignored.
    auto sub = pxr::UsdGeomSubset::Define(stage, pxr::SdfPath("/Mesh/Sub"));
    sub.GetFamilyNameAttr().Set(pxr::TfToken("someOtherFamily"));
    sub.GetElementTypeAttr().Set(pxr::TfToken("face"));
    sub.GetIndicesAttr().Set(pxr::VtIntArray{0, 1});

    pts::rendering::RenderWorld world;
    pts::rendering::populate_from_stage(world, stage);

    // No materialBind subsets -> single whole-mesh object.
    CHECK(count_active_objects(world) == 1);
    CHECK(world.find_object_by_prim(pxr::SdfPath("/Mesh")) >= 0);
}

#endif  // !__EMSCRIPTEN__

PTS_TEST_MAIN()
