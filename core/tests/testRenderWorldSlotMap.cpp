#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/diagnostics.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/vertex.h>
#include <doctest/doctest.h>

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
    CHECK(world.get_objects()[b].active == false);

    auto reused = scope.alloc_object_slot();
    CHECK(reused == b);
    CHECK(world.get_objects()[reused].active == true);
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
    CHECK(world.get_lights()[l0].active == true);

    PTS_UNUSED(a);
    PTS_UNUSED(c);
    PTS_UNUSED(m1);
    PTS_UNUSED(l1);
}

TEST_CASE("find_object_by_prim returns correct index") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_object_slot();
    // SyncScope is a friend — set up prim path via scope.world()
    scope.object(idx).prim_path = "/World/Cube";
    scope.set_prim_slot("/World/Cube", PrimSlot{PrimSlot::Kind::Object, idx});

    CHECK(world.find_object_by_prim("/World/Cube") == static_cast<int>(idx));
}

TEST_CASE("find returns -1 for unknown path") {
    RenderWorld world;
    CHECK(world.find_object_by_prim("/does/not/exist") == -1);
    CHECK(world.find_light_by_prim("/does/not/exist") == -1);
}

TEST_CASE("find_light_by_prim returns correct index") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_light_slot();
    scope.light(idx).prim_path = "/World/Light";
    scope.set_prim_slot("/World/Light", PrimSlot{PrimSlot::Kind::Light, idx});

    CHECK(world.find_light_by_prim("/World/Light") == static_cast<int>(idx));
}

TEST_CASE("free_object_slot removes from prim_slots") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_object_slot();
    scope.object(idx).prim_path = "/World/Sphere";
    scope.set_prim_slot("/World/Sphere", PrimSlot{PrimSlot::Kind::Object, idx});

    scope.free_object_slot(idx);
    CHECK(world.find_object_by_prim("/World/Sphere") == -1);
    CHECK(world.get_objects()[idx].prim_path.empty());
}

TEST_CASE("free_light_slot removes from prim_slots") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_light_slot();
    scope.light(idx).prim_path = "/World/Sun";
    scope.set_prim_slot("/World/Sun", PrimSlot{PrimSlot::Kind::Light, idx});

    scope.free_light_slot(idx);
    CHECK(world.find_light_by_prim("/World/Sun") == -1);
    CHECK(world.get_lights()[idx].active == false);
}

TEST_CASE("clear resets everything") {
    RenderWorld world;
    {
        auto scope = world.begin_sync();

        auto o = scope.alloc_object_slot();
        scope.object(o).prim_path = "/A";
        scope.set_prim_slot("/A", PrimSlot{PrimSlot::Kind::Object, o});

        auto l = scope.alloc_light_slot();
        scope.light(l).prim_path = "/B";
        scope.set_prim_slot("/B", PrimSlot{PrimSlot::Kind::Light, l});

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
    CHECK(world.get_objects()[o].active == true);

    auto l = scope.alloc_light_slot();
    CHECK(world.get_lights()[l].active == true);
}

TEST_CASE("active flag is false after free, true after re-alloc") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto o = scope.alloc_object_slot();
    scope.free_object_slot(o);
    CHECK(world.get_objects()[o].active == false);

    auto o2 = scope.alloc_object_slot();
    CHECK(o2 == o);
    CHECK(world.get_objects()[o2].active == true);

    auto l = scope.alloc_light_slot();
    scope.free_light_slot(l);
    CHECK(world.get_lights()[l].active == false);

    auto l2 = scope.alloc_light_slot();
    CHECK(l2 == l);
    CHECK(world.get_lights()[l2].active == true);
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

TEST_CASE("dirty light tracking") {
    RenderWorld world;

    SUBCASE("alloc marks slot dirty") {
        auto scope = world.begin_sync();
        auto l = scope.alloc_light_slot();
        auto dirty = world.get_dirty_lights();
        REQUIRE(dirty.size() > l);
        CHECK(dirty[l] != 0);
    }

    SUBCASE("clear_dirty_lights resets all bits") {
        auto scope = world.begin_sync();
        scope.alloc_light_slot();
        scope.alloc_light_slot();
        world.clear_dirty_lights();
        auto dirty = world.get_dirty_lights();
        for (std::size_t i = 0; i < dirty.size(); ++i) {
            CHECK(dirty[i] == 0);
        }
    }

    SUBCASE("free marks slot dirty") {
        auto scope = world.begin_sync();
        auto l = scope.alloc_light_slot();
        world.clear_dirty_lights();
        scope.free_light_slot(l);
        auto dirty = world.get_dirty_lights();
        CHECK(dirty[l] != 0);
    }

    SUBCASE("for_each_prim iterates all slots") {
        auto scope = world.begin_sync();
        auto o = scope.alloc_object_slot();
        scope.object(o).prim_path = "/Obj";
        scope.set_prim_slot("/Obj", PrimSlot{PrimSlot::Kind::Object, o});

        auto l = scope.alloc_light_slot();
        scope.light(l).prim_path = "/Light";
        scope.set_prim_slot("/Light", PrimSlot{PrimSlot::Kind::Light, l});

        int count = 0;
        world.for_each_prim([&](std::string_view, PrimSlot) { ++count; });
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

    scope.mesh(m).cpu_vertices = {v};
    scope.mesh(m).cpu_indices = {0};

    CHECK(world.get_meshes()[m].cpu_vertices.size() == 1);
    CHECK(world.get_meshes()[m].cpu_vertices[0].position[0] == doctest::Approx(1.0f));
    CHECK(world.get_meshes()[m].cpu_indices.size() == 1);
}

TEST_CASE("free_mesh_slot clears cpu_vertices") {
    RenderWorld world;
    auto scope = world.begin_sync();
    auto m = scope.alloc_mesh_slot();

    scope.mesh(m).cpu_vertices = {Vertex{}};
    scope.mesh(m).cpu_indices = {0};

    scope.free_mesh_slot(m);
    CHECK(world.get_meshes()[m].cpu_vertices.empty());
    CHECK(world.get_meshes()[m].cpu_indices.empty());
}
