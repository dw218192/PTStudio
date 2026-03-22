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

    CHECK(scope.alloc_object_slot() == 0);
    CHECK(scope.alloc_object_slot() == 1);
    CHECK(scope.alloc_object_slot() == 2);
    CHECK(world.get_objects().size() == 3);

    CHECK(scope.alloc_mesh_slot() == 0);
    CHECK(scope.alloc_mesh_slot() == 1);
    CHECK(world.get_meshes().size() == 2);

    CHECK(scope.alloc_light_slot() == 0);
    CHECK(scope.alloc_light_slot() == 1);
    CHECK(world.get_lights().size() == 2);
}

TEST_CASE("free + re-alloc reuses slots") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto a = scope.alloc_object_slot();
    auto b = scope.alloc_object_slot();
    auto c = scope.alloc_object_slot();

    scope.free_object_slot(b);
    CHECK(world.get_objects()[b].active() == false);

    auto reused = scope.alloc_object_slot();
    CHECK(reused == b);
    CHECK(world.get_objects()[reused].active() == true);
    CHECK(world.get_objects().size() == 3);

    // mesh slot reuse
    auto m0 = scope.alloc_mesh_slot();
    auto m1 = scope.alloc_mesh_slot();
    scope.free_mesh_slot(m0);
    CHECK(scope.alloc_mesh_slot() == m0);

    // light slot reuse
    auto l0 = scope.alloc_light_slot();
    auto l1 = scope.alloc_light_slot();
    scope.free_light_slot(l0);
    CHECK(scope.alloc_light_slot() == l0);
    CHECK(world.get_lights()[l0].active() == true);

    PTS_UNUSED(a);
    PTS_UNUSED(c);
    PTS_UNUSED(m1);
    PTS_UNUSED(l1);
}

TEST_CASE("find_object_by_prim returns correct index") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_object_slot();
    scope.set_prim_path(idx, PrimSlot::Kind::Object, pxr::SdfPath("/World/Cube"));

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

    auto idx = scope.alloc_light_slot();
    scope.set_prim_path(idx, PrimSlot::Kind::Light, pxr::SdfPath("/World/Light"));

    CHECK(world.find_light_by_prim(pxr::SdfPath("/World/Light")) == static_cast<int>(idx));
}

TEST_CASE("free_object_slot removes from prim_slots") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_object_slot();
    scope.set_prim_path(idx, PrimSlot::Kind::Object, pxr::SdfPath("/World/Sphere"));

    scope.free_object_slot(idx);
    CHECK(world.find_object_by_prim(pxr::SdfPath("/World/Sphere")) == -1);
    CHECK(world.get_objects()[idx].get_prim_path().IsEmpty());
}

TEST_CASE("free_light_slot removes from prim_slots") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_light_slot();
    scope.set_prim_path(idx, PrimSlot::Kind::Light, pxr::SdfPath("/World/Sun"));

    scope.free_light_slot(idx);
    CHECK(world.find_light_by_prim(pxr::SdfPath("/World/Sun")) == -1);
    CHECK(world.get_lights()[idx].active() == false);
}

TEST_CASE("clear resets everything") {
    RenderWorld world;
    {
        auto scope = world.begin_sync();

        auto o = scope.alloc_object_slot();
        scope.set_prim_path(o, PrimSlot::Kind::Object, pxr::SdfPath("/A"));

        auto l = scope.alloc_light_slot();
        scope.set_prim_path(l, PrimSlot::Kind::Light, pxr::SdfPath("/B"));

        scope.alloc_mesh_slot();

        scope.free_object_slot(o);
        scope.free_light_slot(l);
    }

    world.clear();

    CHECK(world.get_objects().empty());
    CHECK(world.get_meshes().empty());
    CHECK(world.get_lights().empty());
    CHECK(world.get_materials().empty());
}

TEST_CASE("active flag defaults to true on alloc") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto o = scope.alloc_object_slot();
    CHECK(world.get_objects()[o].active() == true);

    auto l = scope.alloc_light_slot();
    CHECK(world.get_lights()[l].active() == true);
}

TEST_CASE("active flag is false after free, true after re-alloc") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto o = scope.alloc_object_slot();
    scope.free_object_slot(o);
    CHECK(world.get_objects()[o].active() == false);

    auto o2 = scope.alloc_object_slot();
    CHECK(o2 == o);
    CHECK(world.get_objects()[o2].active() == true);

    auto l = scope.alloc_light_slot();
    scope.free_light_slot(l);
    CHECK(world.get_lights()[l].active() == false);

    auto l2 = scope.alloc_light_slot();
    CHECK(l2 == l);
    CHECK(world.get_lights()[l2].active() == true);
}

TEST_CASE("SyncScope bumps mesh_version once") {
    RenderWorld world;
    auto initial = world.get_mesh_version();
    {
        auto scope = world.begin_sync();
        scope.alloc_object_slot();
        scope.alloc_object_slot();
        scope.alloc_mesh_slot();
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

TEST_CASE("generation-based tracking") {
    RenderWorld world;

    SUBCASE("alloc bumps generation") {
        auto scope = world.begin_sync();
        auto l = scope.alloc_light_slot();
        // activate() bumps generation, so it should be > 0
        CHECK(world.get_lights()[l].generation() > 0);
    }

    SUBCASE("write bumps generation") {
        auto scope = world.begin_sync();
        auto l = scope.alloc_light_slot();
        auto gen_before = world.get_lights()[l].generation();
        {
            auto w = scope.write_light(l);
            w->color = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        CHECK(world.get_lights()[l].generation() > gen_before);
    }

    SUBCASE("reused slot has different generation than original") {
        auto scope = world.begin_sync();
        auto l = scope.alloc_light_slot();
        auto gen_after_alloc = world.get_lights()[l].generation();
        scope.free_light_slot(l);
        auto gen_after_free = world.get_lights()[l].generation();
        CHECK(gen_after_free > gen_after_alloc);

        auto l2 = scope.alloc_light_slot();
        CHECK(l2 == l);
        CHECK(world.get_lights()[l2].generation() > gen_after_free);
    }

    SUBCASE("for_each_prim iterates all slots") {
        auto scope = world.begin_sync();
        auto o = scope.alloc_object_slot();
        scope.set_prim_path(o, PrimSlot::Kind::Object, pxr::SdfPath("/Obj"));

        auto l = scope.alloc_light_slot();
        scope.set_prim_path(l, PrimSlot::Kind::Light, pxr::SdfPath("/Light"));

        int count = 0;
        world.for_each_prim([&](const pxr::SdfPath&, PrimSlot) { ++count; });
        CHECK(count == 2);
    }
}

TEST_CASE("Mesh cpu_vertices can be populated via SyncScope") {
    RenderWorld world;
    auto scope = world.begin_sync();
    auto m = scope.alloc_mesh_slot();

    Vertex v{};
    v.position[0] = 1.0f;
    v.position[1] = 2.0f;
    v.position[2] = 3.0f;

    {
        auto w = scope.write_mesh(m);
        w->cpu_vertices = {v};
        w->cpu_indices = {0};
    }

    CHECK(world.get_meshes()[m]->cpu_vertices.size() == 1);
    CHECK(world.get_meshes()[m]->cpu_vertices[0].position[0] == doctest::Approx(1.0f));
    CHECK(world.get_meshes()[m]->cpu_indices.size() == 1);
}

TEST_CASE("free_mesh_slot clears cpu_vertices") {
    RenderWorld world;
    auto scope = world.begin_sync();
    auto m = scope.alloc_mesh_slot();

    {
        auto w = scope.write_mesh(m);
        w->cpu_vertices = {Vertex{}};
        w->cpu_indices = {0};
    }

    scope.free_mesh_slot(m);
    CHECK(world.get_meshes()[m]->cpu_vertices.empty());
    CHECK(world.get_meshes()[m]->cpu_indices.empty());
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
    slot.transform = glm::mat4(1.0f);  // identity

    auto l = to_light(slot);
    // right = normalize(transform[0]) * width/2 = (1,0,0) * 2
    CHECK(l.right.x == doctest::Approx(2.0f));
    CHECK(l.right.y == doctest::Approx(0.0f));
    CHECK(l.right.z == doctest::Approx(0.0f));
    // up = normalize(transform[1]) * height/2 = (0,1,0) * 1
    CHECK(l.up.x == doctest::Approx(0.0f));
    CHECK(l.up.y == doctest::Approx(1.0f));
    CHECK(l.up.z == doctest::Approx(0.0f));
    // position from transform column 3
    CHECK(l.direction_or_pos == glm::vec3(0.0f));
}

TEST_CASE("to_light rect light with rotated transform") {
    LightData slot{};
    slot.type = LightData::Type::Rect;
    slot.width = 6.0f;
    slot.height = 4.0f;
    // 90-degree rotation around Z: X->(0,1,0), Y->(-1,0,0)
    slot.transform = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 0, 1));
    slot.transform[3] = glm::vec4(5.0f, 3.0f, 1.0f, 1.0f);

    auto l = to_light(slot);
    // right = (0,1,0) * 3.0
    CHECK(l.right.x == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(l.right.y == doctest::Approx(3.0f).epsilon(1e-5));
    CHECK(l.right.z == doctest::Approx(0.0f).epsilon(1e-5));
    // up = (-1,0,0) * 2.0
    CHECK(l.up.x == doctest::Approx(-2.0f).epsilon(1e-5));
    CHECK(l.up.y == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(l.up.z == doctest::Approx(0.0f).epsilon(1e-5));
    // position
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
    // right = normalize(transform[0]) * radius = (1,0,0) * 3
    CHECK(l.right.x == doctest::Approx(3.0f));
    CHECK(l.right.y == doctest::Approx(0.0f));
    CHECK(l.right.z == doctest::Approx(0.0f));
    // up = normalize(transform[1]) * radius = (0,1,0) * 3
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

TEST_CASE("Light struct is 64 bytes") {
    CHECK(sizeof(Light) == 64);
}
