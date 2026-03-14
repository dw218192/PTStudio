#include <core/rendering/adapters/capsuleAdapter.h>
#include <core/rendering/adapters/coneAdapter.h>
#include <core/rendering/adapters/cubeAdapter.h>
#include <core/rendering/adapters/cylinderAdapter.h>
#include <core/rendering/adapters/registry.h>
#include <core/rendering/adapters/sphereAdapter.h>
#include <embedded_test_resources.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/sphere.h>

#include <cmath>
#include <string>
#include <variant>

#include "testApplication.h"

namespace {

// Verify all normals are unit length (within tolerance).
void check_normals_normalized(const std::vector<pts::rendering::Vertex>& vertices) {
    for (size_t i = 0; i < vertices.size(); ++i) {
        const auto& v = vertices[i];
        float len = std::sqrt(v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] +
                              v.normal[2] * v.normal[2]);
        CHECK_MESSAGE(len == doctest::Approx(1.0f).epsilon(0.01f), "normal at vertex ", i,
                      " has length ", len);
    }
}

// Verify index count is a multiple of 3 (triangulated).
void check_triangulated(const std::vector<uint32_t>& indices) {
    CHECK(indices.size() % 3 == 0);
}

// Verify all indices are in range.
void check_indices_valid(const std::vector<uint32_t>& indices, size_t vertex_count) {
    for (size_t i = 0; i < indices.size(); ++i) {
        CHECK_MESSAGE(indices[i] < vertex_count, "index ", i, " = ", indices[i], " out of range (",
                      vertex_count, " vertices)");
    }
}

const pts::rendering::MeshResult& get_mesh(
    const std::optional<pts::rendering::AdapterResult>& result) {
    REQUIRE(result.has_value());
    auto* mesh = std::get_if<pts::rendering::MeshResult>(&*result);
    REQUIRE(mesh);
    return *mesh;
}

void validate_result(const std::optional<pts::rendering::AdapterResult>& result) {
    const auto& mesh = get_mesh(result);
    CHECK(!mesh.vertices.empty());
    CHECK(!mesh.indices.empty());
    check_triangulated(mesh.indices);
    check_indices_valid(mesh.indices, mesh.vertices.size());
    check_normals_normalized(mesh.vertices);
}

}  // namespace

TEST_CASE("CubeAdapter - basic cube") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(2.0);

    const auto& adapter = pts::rendering::CubeAdapter::instance();
    CHECK(adapter.can_adapt(cube.GetPrim()));

    auto result = adapter.adapt(cube.GetPrim());
    validate_result(result);

    // 24 vertices (4 per face x 6 faces), 36 indices (2 tris per face x 6 faces)
    CHECK(get_mesh(result).vertices.size() == 24);
    CHECK(get_mesh(result).indices.size() == 36);
}

TEST_CASE("CubeAdapter - respects size attribute") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(4.0);

    auto result = pts::rendering::CubeAdapter::instance().adapt(cube.GetPrim());
    REQUIRE(result.has_value());

    // With size=4, half-extent=2. Check that some vertex reaches 2.0.
    bool found_extent = false;
    for (const auto& v : get_mesh(result).vertices) {
        if (std::abs(v.position[0]) == doctest::Approx(2.0f) ||
            std::abs(v.position[1]) == doctest::Approx(2.0f) ||
            std::abs(v.position[2]) == doctest::Approx(2.0f)) {
            found_extent = true;
            break;
        }
    }
    CHECK(found_extent);
}

TEST_CASE("CubeAdapter - displayColor primvar") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(2.0);

    auto primvars_api = pxr::UsdGeomPrimvarsAPI(cube.GetPrim());
    auto color_pv = primvars_api.CreatePrimvar(pxr::TfToken("displayColor"),
                                               pxr::SdfValueTypeNames->Color3fArray);
    pxr::VtVec3fArray colors = {{1.0f, 0.0f, 0.0f}};
    color_pv.Set(colors);

    auto result = pts::rendering::CubeAdapter::instance().adapt(cube.GetPrim());
    REQUIRE(result.has_value());

    for (const auto& v : get_mesh(result).vertices) {
        CHECK(v.color[0] == doctest::Approx(1.0f));
        CHECK(v.color[1] == doctest::Approx(0.0f));
        CHECK(v.color[2] == doctest::Approx(0.0f));
    }
}

TEST_CASE("SphereAdapter - basic sphere") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto sphere = pxr::UsdGeomSphere::Define(stage, pxr::SdfPath("/Sphere"));
    sphere.GetRadiusAttr().Set(1.0);

    const auto& adapter = pts::rendering::SphereAdapter::instance();
    CHECK(adapter.can_adapt(sphere.GetPrim()));

    auto result = adapter.adapt(sphere.GetPrim());
    validate_result(result);

    // UV-sphere with 16 lat x 32 lon: (16+1)*(32+1) = 561 vertices
    CHECK(get_mesh(result).vertices.size() == 561);
}

TEST_CASE("SphereAdapter - respects radius") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto sphere = pxr::UsdGeomSphere::Define(stage, pxr::SdfPath("/Sphere"));
    sphere.GetRadiusAttr().Set(3.0);

    auto result = pts::rendering::SphereAdapter::instance().adapt(sphere.GetPrim());
    REQUIRE(result.has_value());

    // Check that all vertices are at distance ~3 from origin
    for (const auto& v : get_mesh(result).vertices) {
        float dist = std::sqrt(v.position[0] * v.position[0] + v.position[1] * v.position[1] +
                               v.position[2] * v.position[2]);
        CHECK(dist == doctest::Approx(3.0f).epsilon(0.01f));
    }
}

TEST_CASE("CylinderAdapter - basic cylinder") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cyl = pxr::UsdGeomCylinder::Define(stage, pxr::SdfPath("/Cylinder"));

    const auto& adapter = pts::rendering::CylinderAdapter::instance();
    CHECK(adapter.can_adapt(cyl.GetPrim()));

    auto result = adapter.adapt(cyl.GetPrim());
    validate_result(result);

    // Must have at least side + 2 caps worth of triangles
    CHECK(get_mesh(result).indices.size() > 0);
}

TEST_CASE("CylinderAdapter - axis attribute") {
    auto stage = pxr::UsdStage::CreateInMemory();

    SUBCASE("Z axis") {
        auto cyl = pxr::UsdGeomCylinder::Define(stage, pxr::SdfPath("/CylZ"));
        cyl.GetAxisAttr().Set(pxr::UsdGeomTokens->z);
        cyl.GetHeightAttr().Set(4.0);

        auto result = pts::rendering::CylinderAdapter::instance().adapt(cyl.GetPrim());
        REQUIRE(result.has_value());

        // With Z axis and height 4, half_h=2. Cap center should be at z=+/-2.
        bool found_top = false;
        for (const auto& v : get_mesh(result).vertices) {
            if (v.position[2] == doctest::Approx(2.0f)) {
                found_top = true;
                break;
            }
        }
        CHECK(found_top);
    }

    SUBCASE("X axis") {
        auto cyl = pxr::UsdGeomCylinder::Define(stage, pxr::SdfPath("/CylX"));
        cyl.GetAxisAttr().Set(pxr::UsdGeomTokens->x);
        cyl.GetHeightAttr().Set(4.0);

        auto result = pts::rendering::CylinderAdapter::instance().adapt(cyl.GetPrim());
        REQUIRE(result.has_value());

        bool found_top = false;
        for (const auto& v : get_mesh(result).vertices) {
            if (v.position[0] == doctest::Approx(2.0f)) {
                found_top = true;
                break;
            }
        }
        CHECK(found_top);
    }
}

TEST_CASE("ConeAdapter - basic cone") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cone = pxr::UsdGeomCone::Define(stage, pxr::SdfPath("/Cone"));

    const auto& adapter = pts::rendering::ConeAdapter::instance();
    CHECK(adapter.can_adapt(cone.GetPrim()));

    auto result = adapter.adapt(cone.GetPrim());
    validate_result(result);
    CHECK(get_mesh(result).indices.size() > 0);
}

TEST_CASE("ConeAdapter - apex at correct position") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cone = pxr::UsdGeomCone::Define(stage, pxr::SdfPath("/Cone"));

    // Read back the height to verify what USD gives us
    double height_d = 0;
    cone.GetHeightAttr().Get(&height_d);
    float expected_apex_y = static_cast<float>(height_d) * 0.5f;
    REQUIRE(expected_apex_y > 0.0f);

    auto result = pts::rendering::ConeAdapter::instance().adapt(cone.GetPrim());
    REQUIRE(result.has_value());

    // Default Y axis → apex at y = +half_height
    bool found_apex = false;
    for (const auto& v : get_mesh(result).vertices) {
        if (v.position[1] == doctest::Approx(expected_apex_y)) {
            found_apex = true;
            break;
        }
    }
    CHECK(found_apex);
}

TEST_CASE("CapsuleAdapter - basic capsule") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cap = pxr::UsdGeomCapsule::Define(stage, pxr::SdfPath("/Capsule"));

    const auto& adapter = pts::rendering::CapsuleAdapter::instance();
    CHECK(adapter.can_adapt(cap.GetPrim()));

    auto result = adapter.adapt(cap.GetPrim());
    validate_result(result);
    CHECK(get_mesh(result).indices.size() > 0);
}

TEST_CASE("CapsuleAdapter - hemisphere extends beyond cylinder height") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cap = pxr::UsdGeomCapsule::Define(stage, pxr::SdfPath("/Capsule"));

    // Read back the actual USD attribute values
    double radius_d = 0, height_d = 0;
    cap.GetRadiusAttr().Get(&radius_d);
    cap.GetHeightAttr().Get(&height_d);
    float half_h = static_cast<float>(height_d) * 0.5f;

    auto result = pts::rendering::CapsuleAdapter::instance().adapt(cap.GetPrim());
    REQUIRE(result.has_value());

    // USD default axis for capsule is Z, so check max along Z
    float max_along = -1e9f;
    for (const auto& v : get_mesh(result).vertices) {
        for (int a = 0; a < 3; ++a) {
            if (v.position[a] > max_along) max_along = v.position[a];
        }
    }
    // Hemisphere pole should be at half_h + radius
    float expected_max = half_h + static_cast<float>(radius_d);
    CHECK(max_along == doctest::Approx(expected_max).epsilon(0.01f));
}

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

    // Walk registry — at least one adapter must handle this prim
    bool adapted = false;
    for (const auto* adapter : pts::rendering::k_schema_adapters()) {
        if (!adapter->can_adapt(cube_prim)) continue;
        auto result = adapter->adapt(cube_prim);
        REQUIRE(result.has_value());
        CHECK(!get_mesh(result).vertices.empty());
        CHECK(!get_mesh(result).indices.empty());
        check_normals_normalized(get_mesh(result).vertices);
        adapted = true;
        break;
    }
    CHECK(adapted);
}

PTS_TEST_MAIN()
