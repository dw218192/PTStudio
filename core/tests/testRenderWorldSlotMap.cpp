#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/diagnostics.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/vertex.h>
#include <doctest/doctest.h>
#include <pxr/usd/sdf/path.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace pts::rendering;

TEST_CASE("alloc returns sequential indices on empty world") {
    RenderWorld world;
    auto scope = world.begin_sync();

    CHECK(scope.alloc_object(pxr::SdfPath("/A")) == 0);
    CHECK(scope.alloc_object(pxr::SdfPath("/B")) == 1);
    CHECK(scope.alloc_object(pxr::SdfPath("/C")) == 2);
    CHECK(world.get_objects().size() == 3);

    CHECK(scope.alloc_mesh(pxr::SdfPath("/M0")) == 0);
    CHECK(scope.alloc_mesh(pxr::SdfPath("/M1")) == 1);
    CHECK(world.get_meshes().size() == 2);

    CHECK(scope.alloc_light(pxr::SdfPath("/L0")) == 0);
    CHECK(scope.alloc_light(pxr::SdfPath("/L1")) == 1);
    CHECK(world.get_lights().size() == 2);
}

TEST_CASE("free + re-alloc reuses slots") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto a = scope.alloc_object(pxr::SdfPath("/A"));
    auto b = scope.alloc_object(pxr::SdfPath("/B"));
    auto c = scope.alloc_object(pxr::SdfPath("/C"));

    scope.free_object(pxr::SdfPath("/B"));
    CHECK(!world.get_objects().active_at(b));

    auto reused = scope.alloc_object(pxr::SdfPath("/D"));
    CHECK(reused == b);
    CHECK(world.get_objects().active_at(reused));
    CHECK(world.get_objects().capacity() == 3);

    // mesh slot reuse
    auto m0 = scope.alloc_mesh(pxr::SdfPath("/M0"));
    auto m1 = scope.alloc_mesh(pxr::SdfPath("/M1"));
    scope.free_mesh(pxr::SdfPath("/M0"));
    CHECK(scope.alloc_mesh(pxr::SdfPath("/M2")) == m0);

    // light slot reuse
    auto l0 = scope.alloc_light(pxr::SdfPath("/L0"));
    auto l1 = scope.alloc_light(pxr::SdfPath("/L1"));
    scope.free_light(pxr::SdfPath("/L0"));
    auto l_reused = scope.alloc_light(pxr::SdfPath("/L2"));
    CHECK(l_reused == l0);
    CHECK(world.get_lights().active_at(l_reused));

    UNUSED(a);
    UNUSED(c);
    UNUSED(m1);
    UNUSED(l1);
}

TEST_CASE("find_object_by_prim returns correct index") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_object(pxr::SdfPath("/World/Cube"));

    CHECK(world.find_object_by_prim(pxr::SdfPath("/World/Cube")) == static_cast<int>(idx));
}

TEST_CASE("find returns -1 for unknown path") {
    RenderWorld world;
    CHECK(world.find_object_by_prim(pxr::SdfPath("/does/not/exist")) == -1);
    CHECK(world.find_light_by_prim(pxr::SdfPath("/does/not/exist")) == -1);
}

TEST_CASE("find_light_by_prim returns correct index") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_light(pxr::SdfPath("/World/Light"));

    CHECK(world.find_light_by_prim(pxr::SdfPath("/World/Light")) == static_cast<int>(idx));
}

TEST_CASE("free_object removes from lookup") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_object(pxr::SdfPath("/World/Sphere"));

    scope.free_object(pxr::SdfPath("/World/Sphere"));
    CHECK(world.find_object_by_prim(pxr::SdfPath("/World/Sphere")) == -1);
    CHECK(!world.get_objects().active_at(idx));
}

TEST_CASE("free_light removes from lookup") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_light(pxr::SdfPath("/World/Sun"));

    scope.free_light(pxr::SdfPath("/World/Sun"));
    CHECK(world.find_light_by_prim(pxr::SdfPath("/World/Sun")) == -1);
    CHECK(!world.get_lights().active_at(idx));
}

TEST_CASE("clear resets everything") {
    RenderWorld world;
    {
        auto scope = world.begin_sync();

        scope.alloc_object(pxr::SdfPath("/A"));
        scope.alloc_light(pxr::SdfPath("/B"));
        scope.alloc_mesh(pxr::SdfPath("/M"));

        scope.free_object(pxr::SdfPath("/A"));
        scope.free_light(pxr::SdfPath("/B"));
    }

    world.clear();

    CHECK(world.get_objects().size() == 0);
    CHECK(world.get_meshes().size() == 0);
    CHECK(world.get_lights().size() == 0);
    CHECK(world.get_materials().empty());
}

TEST_CASE("active flag is false after free, true after re-alloc") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto o = scope.alloc_object(pxr::SdfPath("/O"));
    scope.free_object(pxr::SdfPath("/O"));
    CHECK(!world.get_objects().active_at(o));

    auto o2 = scope.alloc_object(pxr::SdfPath("/O2"));
    CHECK(o2 == o);
    CHECK(world.get_objects().active_at(o2));

    auto l = scope.alloc_light(pxr::SdfPath("/L"));
    scope.free_light(pxr::SdfPath("/L"));
    CHECK(!world.get_lights().active_at(l));

    auto l2 = scope.alloc_light(pxr::SdfPath("/L2"));
    CHECK(l2 == l);
    CHECK(world.get_lights().active_at(l2));
}

TEST_CASE("SyncScope bumps mesh_version once") {
    RenderWorld world;
    auto initial = world.get_mesh_version();
    {
        auto scope = world.begin_sync();
        scope.alloc_object(pxr::SdfPath("/A"));
        scope.alloc_object(pxr::SdfPath("/B"));
        scope.alloc_mesh(pxr::SdfPath("/M"));
    }
    CHECK(world.get_mesh_version() == initial + 1);
}

TEST_CASE("SyncScope bumps material_version once") {
    RenderWorld world;
    auto initial = world.get_material_version();
    {
        auto scope = world.begin_sync();
        scope.materials().push_back(Material{});
    }
    CHECK(world.get_material_version() == initial + 1);
}

TEST_CASE("version-based tracking") {
    RenderWorld world;

    SUBCASE("alloc bumps version") {
        auto scope = world.begin_sync();
        auto l = scope.alloc_light(pxr::SdfPath("/L"));
        CHECK(world.get_lights().version_at(l) > 0);
    }

    SUBCASE("mutate bumps version") {
        auto scope = world.begin_sync();
        auto l = scope.alloc_light(pxr::SdfPath("/L"));
        auto ver_before = world.get_lights().version_at(l);
        scope.mutate_light(l, [](LightData& ld) { ld.color = glm::vec3(1.0f, 0.0f, 0.0f); });
        CHECK(world.get_lights().version_at(l) > ver_before);
    }

    SUBCASE("for_each_prim iterates all slots") {
        auto scope = world.begin_sync();
        scope.alloc_object(pxr::SdfPath("/Obj"));
        scope.alloc_light(pxr::SdfPath("/Light"));

        int count = 0;
        world.for_each_prim([&](const pxr::SdfPath&, PrimSlot) { ++count; });
        CHECK(count == 2);
    }
}

TEST_CASE("Mesh cpu_vertices can be populated via SyncScope") {
    RenderWorld world;
    auto scope = world.begin_sync();
    auto m = scope.alloc_mesh(pxr::SdfPath("/Mesh"));

    Vertex v{};
    v.position[0] = 1.0f;
    v.position[1] = 2.0f;
    v.position[2] = 3.0f;

    scope.mutate_mesh(m, [&](MeshData& mesh) {
        mesh.cpu_vertices = {v};
        mesh.cpu_indices = {0};
    });

    CHECK(world.get_meshes().at(m).cpu_vertices.size() == 1);
    CHECK(world.get_meshes().at(m).cpu_vertices[0].position[0] == doctest::Approx(1.0f));
    CHECK(world.get_meshes().at(m).cpu_indices.size() == 1);
}

TEST_CASE("free_mesh clears mesh data") {
    RenderWorld world;
    auto scope = world.begin_sync();
    auto m = scope.alloc_mesh(pxr::SdfPath("/Mesh"));

    scope.mutate_mesh(m, [](MeshData& mesh) {
        mesh.cpu_vertices = {Vertex{}};
        mesh.cpu_indices = {0};
    });

    scope.free_mesh(pxr::SdfPath("/Mesh"));
    // After erase, the entry value is reset to default
    auto raw = world.get_meshes().span_raw();
    CHECK(raw[m].value.cpu_vertices.empty());
    CHECK(raw[m].value.cpu_indices.empty());
}

// --- to_light() orientation vector tests ---

TEST_CASE("to_light distant light has zero right/up") {
    LightData slot{};
    slot.type = LightData::Type::Distant;
    slot.direction = glm::vec3(0.0f, -1.0f, 0.0f);
    slot.color = glm::vec3(1.0f);
    slot.intensity = 2.0f;

    auto l = to_light(slot);
    CHECK(l.right == glm::vec3(0.0f));
    CHECK(l.up == glm::vec3(0.0f));
    CHECK(l.direction_or_pos == slot.direction);
}

TEST_CASE("to_light rect light encodes half-size orientation vectors") {
    LightData slot{};
    slot.type = LightData::Type::Rect;
    slot.width = 4.0f;
    slot.height = 2.0f;
    slot.transform = glm::mat4(1.0f);

    auto l = to_light(slot);
    CHECK(l.right.x == doctest::Approx(2.0f));
    CHECK(l.right.y == doctest::Approx(0.0f));
    CHECK(l.right.z == doctest::Approx(0.0f));
    CHECK(l.up.x == doctest::Approx(0.0f));
    CHECK(l.up.y == doctest::Approx(1.0f));
    CHECK(l.up.z == doctest::Approx(0.0f));
    CHECK(l.direction_or_pos == glm::vec3(0.0f));
}

TEST_CASE("to_light rect light with rotated transform") {
    LightData slot{};
    slot.type = LightData::Type::Rect;
    slot.width = 6.0f;
    slot.height = 4.0f;
    slot.transform = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 0, 1));
    slot.transform[3] = glm::vec4(5.0f, 3.0f, 1.0f, 1.0f);

    auto l = to_light(slot);
    CHECK(l.right.x == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(l.right.y == doctest::Approx(3.0f).epsilon(1e-5));
    CHECK(l.right.z == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(l.up.x == doctest::Approx(-2.0f).epsilon(1e-5));
    CHECK(l.up.y == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(l.up.z == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(l.direction_or_pos.x == doctest::Approx(5.0f));
    CHECK(l.direction_or_pos.y == doctest::Approx(3.0f));
    CHECK(l.direction_or_pos.z == doctest::Approx(1.0f));
}

TEST_CASE("to_light disk light encodes radius-scaled orientation vectors") {
    LightData slot{};
    slot.type = LightData::Type::Disk;
    slot.radius = 3.0f;
    slot.transform = glm::mat4(1.0f);

    auto l = to_light(slot);
    CHECK(l.right.x == doctest::Approx(3.0f));
    CHECK(l.right.y == doctest::Approx(0.0f));
    CHECK(l.right.z == doctest::Approx(0.0f));
    CHECK(l.up.x == doctest::Approx(0.0f));
    CHECK(l.up.y == doctest::Approx(3.0f));
    CHECK(l.up.z == doctest::Approx(0.0f));
}

TEST_CASE("to_light sphere light has zero right/up") {
    LightData slot{};
    slot.type = LightData::Type::Sphere;
    slot.radius = 1.5f;
    slot.transform = glm::mat4(1.0f);
    slot.transform[3] = glm::vec4(1.0f, 2.0f, 3.0f, 1.0f);

    auto l = to_light(slot);
    CHECK(l.right == glm::vec3(0.0f));
    CHECK(l.up == glm::vec3(0.0f));
    CHECK(l.radius == doctest::Approx(1.5f));
    CHECK(l.direction_or_pos == glm::vec3(1.0f, 2.0f, 3.0f));
}
