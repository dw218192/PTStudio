#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/diagnostics.h>
#include <core/rendering/adapterHelpers.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <core/rendering/vertex.h>
#include <doctest/doctest.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>

#include <cmath>
#include <glm/glm.hpp>

using namespace pts::rendering;

// --- Geometry generator tests ---

TEST_CASE("generate_rect_mesh produces a quad") {
    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
    generate_rect_mesh(4.0f, 2.0f, verts, indices);

    CHECK(verts.size() == 4);
    CHECK(indices.size() == 6);

    // All normals face -Z (USD lights emit along -Z)
    for (auto& v : verts) {
        CHECK(v.normal[0] == doctest::Approx(0.0f));
        CHECK(v.normal[1] == doctest::Approx(0.0f));
        CHECK(v.normal[2] == doctest::Approx(-1.0f));
        // White color
        CHECK(v.color[0] == doctest::Approx(1.0f));
        CHECK(v.color[1] == doctest::Approx(1.0f));
        CHECK(v.color[2] == doctest::Approx(1.0f));
    }

    // Extents: half-width=2, half-height=1
    float min_x = verts[0].position[0], max_x = verts[0].position[0];
    float min_y = verts[0].position[1], max_y = verts[0].position[1];
    for (auto& v : verts) {
        min_x = std::min(min_x, v.position[0]);
        max_x = std::max(max_x, v.position[0]);
        min_y = std::min(min_y, v.position[1]);
        max_y = std::max(max_y, v.position[1]);
        CHECK(v.position[2] == doctest::Approx(0.0f));
    }
    CHECK(min_x == doctest::Approx(-2.0f));
    CHECK(max_x == doctest::Approx(2.0f));
    CHECK(min_y == doctest::Approx(-1.0f));
    CHECK(max_y == doctest::Approx(1.0f));

    // UVs cover 0-1
    bool has_uv00 = false, has_uv11 = false;
    for (auto& v : verts) {
        if (v.uv[0] == doctest::Approx(0.0f) && v.uv[1] == doctest::Approx(0.0f)) has_uv00 = true;
        if (v.uv[0] == doctest::Approx(1.0f) && v.uv[1] == doctest::Approx(1.0f)) has_uv11 = true;
    }
    CHECK(has_uv00);
    CHECK(has_uv11);
}

TEST_CASE("generate_disk_mesh produces triangle fan") {
    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
    generate_disk_mesh(3.0f, verts, indices);

    // Center + 49 rim vertices (48 segments + 1 to close)
    CHECK(verts.size() == 50);
    // 48 triangles * 3 indices
    CHECK(indices.size() == 144);

    // Center vertex at origin
    CHECK(verts[0].position[0] == doctest::Approx(0.0f));
    CHECK(verts[0].position[1] == doctest::Approx(0.0f));
    CHECK(verts[0].position[2] == doctest::Approx(0.0f));

    // All normals face -Z (USD lights emit along -Z)
    for (auto& v : verts) {
        CHECK(v.normal[2] == doctest::Approx(-1.0f));
    }

    // Rim vertices at radius distance
    for (size_t i = 1; i < verts.size(); ++i) {
        float dist = std::sqrt(verts[i].position[0] * verts[i].position[0] +
                               verts[i].position[1] * verts[i].position[1]);
        CHECK(dist == doctest::Approx(3.0f).epsilon(1e-5));
    }

    // All fan triangles start with center (index 0)
    for (size_t i = 0; i < indices.size(); i += 3) {
        CHECK(indices[i] == 0);
    }
}

TEST_CASE("generate_sphere_mesh produces UV sphere") {
    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
    generate_sphere_mesh(2.0f, verts, indices);

    // (k_lat+1) * (k_lon+1) = 9 * 17 = 153 verts
    CHECK(verts.size() == 153);
    // k_lat * k_lon * 6 = 8 * 16 * 6 = 768 indices
    CHECK(indices.size() == 768);

    // All vertices at radius distance
    for (auto& v : verts) {
        float dist = std::sqrt(v.position[0] * v.position[0] + v.position[1] * v.position[1] +
                               v.position[2] * v.position[2]);
        CHECK(dist == doctest::Approx(2.0f).epsilon(1e-4));
    }

    // Normals are unit-length and point outward (same direction as position)
    for (auto& v : verts) {
        float nlen = std::sqrt(v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] +
                               v.normal[2] * v.normal[2]);
        CHECK(nlen == doctest::Approx(1.0f).epsilon(1e-4));
    }
}

// --- sync_light integration tests ---

TEST_CASE("sync_light creates proxy mesh for rect light") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto prim = pxr::UsdLuxRectLight::Define(stage, pxr::SdfPath("/Light")).GetPrim();

    RenderWorld world;
    auto scope = world.begin_sync();

    LightData light;
    light.type = LightData::Type::Rect;
    light.width = 2.0f;
    light.height = 1.0f;
    light.color = {1, 0.5f, 0};
    light.intensity = 5.0f;

    sync_light(prim, scope, light);

    int idx = world.find_light_by_prim(pxr::SdfPath("/Light"));
    REQUIRE(idx >= 0);
    const auto& ld = scope.light(static_cast<uint32_t>(idx));
    CHECK(ld.mesh_index != UINT32_MAX);
    CHECK(ld.material_index != k_no_material);

    // Mesh has data
    const auto& mesh = scope.mesh(ld.mesh_index);
    CHECK(mesh.cpu_vertices.size() == 4);
    CHECK(mesh.cpu_indices.size() == 6);

    // Material is emissive
    auto& mat = scope.materials()[ld.material_index];
    CHECK(mat.diffuse_color == glm::vec3(0, 0, 0));
    CHECK(mat.emissive_color.r == doctest::Approx(5.0f));
    CHECK(mat.emissive_color.g == doctest::Approx(2.5f));
    CHECK(mat.emissive_color.b == doctest::Approx(0.0f));
}

TEST_CASE("sync_light creates proxy mesh for disk light") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto prim = pxr::UsdLuxDiskLight::Define(stage, pxr::SdfPath("/Disk")).GetPrim();

    RenderWorld world;
    auto scope = world.begin_sync();

    LightData light;
    light.type = LightData::Type::Disk;
    light.radius = 1.5f;
    light.color = {1, 1, 1};
    light.intensity = 2.0f;

    sync_light(prim, scope, light);

    int idx = world.find_light_by_prim(pxr::SdfPath("/Disk"));
    REQUIRE(idx >= 0);
    const auto& ld = scope.light(static_cast<uint32_t>(idx));
    CHECK(ld.mesh_index != UINT32_MAX);
    CHECK(scope.mesh(ld.mesh_index).cpu_vertices.size() == 50);
}

TEST_CASE("sync_light creates proxy mesh for sphere light") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto prim = pxr::UsdLuxSphereLight::Define(stage, pxr::SdfPath("/Sphere")).GetPrim();

    RenderWorld world;
    auto scope = world.begin_sync();

    LightData light;
    light.type = LightData::Type::Sphere;
    light.radius = 1.0f;
    light.color = {1, 1, 1};
    light.intensity = 1.0f;

    sync_light(prim, scope, light);

    int idx = world.find_light_by_prim(pxr::SdfPath("/Sphere"));
    REQUIRE(idx >= 0);
    const auto& ld = scope.light(static_cast<uint32_t>(idx));
    CHECK(ld.mesh_index != UINT32_MAX);
    CHECK(scope.mesh(ld.mesh_index).cpu_vertices.size() == 153);
}

TEST_CASE("sync_light does NOT create proxy mesh for distant light") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto prim = pxr::UsdLuxDistantLight::Define(stage, pxr::SdfPath("/Sun")).GetPrim();

    RenderWorld world;
    auto scope = world.begin_sync();

    LightData light;
    light.type = LightData::Type::Distant;

    sync_light(prim, scope, light);

    int idx = world.find_light_by_prim(pxr::SdfPath("/Sun"));
    REQUIRE(idx >= 0);
    const auto& ld = scope.light(static_cast<uint32_t>(idx));
    CHECK(ld.mesh_index == UINT32_MAX);
    CHECK(ld.material_index == k_no_material);
}

TEST_CASE("sync_light re-sync updates geometry and material in place") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto prim = pxr::UsdLuxRectLight::Define(stage, pxr::SdfPath("/Rect")).GetPrim();

    RenderWorld world;
    auto scope = world.begin_sync();

    LightData light;
    light.type = LightData::Type::Rect;
    light.width = 2.0f;
    light.height = 1.0f;
    light.color = {1, 0, 0};
    light.intensity = 1.0f;

    sync_light(prim, scope, light);

    int idx = world.find_light_by_prim(pxr::SdfPath("/Rect"));
    REQUIRE(idx >= 0);
    auto mesh_idx = scope.light(static_cast<uint32_t>(idx)).mesh_index;
    auto mat_idx = scope.light(static_cast<uint32_t>(idx)).material_index;

    // Re-sync with different dimensions and color
    light.width = 4.0f;
    light.height = 3.0f;
    light.color = {0, 1, 0};
    light.intensity = 3.0f;
    sync_light(prim, scope, light);

    // Mesh slot reused
    CHECK(scope.light(static_cast<uint32_t>(idx)).mesh_index == mesh_idx);
    // Material index reused (same cache key)
    CHECK(scope.light(static_cast<uint32_t>(idx)).material_index == mat_idx);

    // Material emissive updated
    auto& mat = scope.materials()[mat_idx];
    CHECK(mat.emissive_color.r == doctest::Approx(0.0f));
    CHECK(mat.emissive_color.g == doctest::Approx(3.0f));
    CHECK(mat.emissive_color.b == doctest::Approx(0.0f));

    // Geometry updated: verify vertices reflect the new dimensions
    const auto& mesh_data = scope.mesh(mesh_idx);
    REQUIRE(!mesh_data.cpu_vertices.empty());
    float max_x = 0.0f;
    for (const auto& v : mesh_data.cpu_vertices) max_x = std::max(max_x, std::abs(v.position[0]));
    CHECK(max_x == doctest::Approx(2.0f));  // half of new width (4.0)
}

TEST_CASE("remove_prim frees proxy mesh slot for lights") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto prim = pxr::UsdLuxRectLight::Define(stage, pxr::SdfPath("/Light")).GetPrim();

    RenderWorld world;
    {
        auto scope = world.begin_sync();

        LightData light;
        light.type = LightData::Type::Rect;
        light.width = 1.0f;
        light.height = 1.0f;
        light.color = {1, 1, 1};
        light.intensity = 1.0f;

        sync_light(prim, scope, light);

        int idx = world.find_light_by_prim(pxr::SdfPath("/Light"));
        REQUIRE(idx >= 0);
        auto mesh_idx = scope.light(static_cast<uint32_t>(idx)).mesh_index;
        REQUIRE(mesh_idx != UINT32_MAX);

        // Remove it
        remove_prim(scope, pxr::SdfPath("/Light"));

        CHECK(world.find_light_by_prim(pxr::SdfPath("/Light")) == -1);
        // Mesh slot freed (inactive)
        CHECK(world.get_meshes().active_at(mesh_idx) == false);
    }
}
