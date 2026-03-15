#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/rendering/renderWorld.h>
#include <doctest/doctest.h>

using namespace pts::rendering;

TEST_CASE("alloc returns sequential indices on empty world") {
    RenderWorld world;
    auto scope = world.begin_sync();

    CHECK(scope.alloc_object_slot() == 0);
    CHECK(scope.alloc_object_slot() == 1);
    CHECK(scope.alloc_object_slot() == 2);
    CHECK(world.objects.size() == 3);

    CHECK(scope.alloc_mesh_slot() == 0);
    CHECK(scope.alloc_mesh_slot() == 1);
    CHECK(world.meshes.size() == 2);

    CHECK(scope.alloc_light_slot() == 0);
    CHECK(scope.alloc_light_slot() == 1);
    CHECK(world.lights.size() == 2);
}

TEST_CASE("free + re-alloc reuses slots") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto a = scope.alloc_object_slot();
    auto b = scope.alloc_object_slot();
    auto c = scope.alloc_object_slot();

    scope.free_object_slot(b);
    CHECK(world.objects[b].active == false);

    auto reused = scope.alloc_object_slot();
    CHECK(reused == b);
    CHECK(world.objects[reused].active == true);
    CHECK(world.objects.size() == 3);

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
    CHECK(world.lights[l0].active == true);

    (void) a;
    (void) c;
    (void) m1;
    (void) l1;
}

TEST_CASE("find_object_by_prim returns correct index") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_object_slot();
    world.objects[idx].prim_path = "/World/Cube";
    world.prim_slots["/World/Cube"] = PrimSlot{PrimSlot::Kind::Object, idx};

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
    world.lights[idx].prim_path = "/World/Light";
    world.prim_slots["/World/Light"] = PrimSlot{PrimSlot::Kind::Light, idx};

    CHECK(world.find_light_by_prim("/World/Light") == static_cast<int>(idx));
}

TEST_CASE("free_object_slot removes from prim_slots") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_object_slot();
    world.objects[idx].prim_path = "/World/Sphere";
    world.prim_slots["/World/Sphere"] = PrimSlot{PrimSlot::Kind::Object, idx};

    scope.free_object_slot(idx);
    CHECK(world.find_object_by_prim("/World/Sphere") == -1);
    CHECK(world.objects[idx].prim_path.empty());
}

TEST_CASE("free_light_slot removes from prim_slots") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto idx = scope.alloc_light_slot();
    world.lights[idx].prim_path = "/World/Sun";
    world.prim_slots["/World/Sun"] = PrimSlot{PrimSlot::Kind::Light, idx};

    scope.free_light_slot(idx);
    CHECK(world.find_light_by_prim("/World/Sun") == -1);
    CHECK(world.lights[idx].active == false);
}

TEST_CASE("clear resets everything") {
    RenderWorld world;
    {
        auto scope = world.begin_sync();

        auto o = scope.alloc_object_slot();
        world.objects[o].prim_path = "/A";
        world.prim_slots["/A"] = PrimSlot{PrimSlot::Kind::Object, o};

        auto l = scope.alloc_light_slot();
        world.lights[l].prim_path = "/B";
        world.prim_slots["/B"] = PrimSlot{PrimSlot::Kind::Light, l};

        scope.alloc_mesh_slot();

        scope.free_object_slot(o);
        scope.free_light_slot(l);
    }

    world.clear();

    CHECK(world.objects.empty());
    CHECK(world.meshes.empty());
    CHECK(world.lights.empty());
    CHECK(world.materials.empty());
    CHECK(world.prim_slots.empty());
}

TEST_CASE("active flag defaults to true on alloc") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto o = scope.alloc_object_slot();
    CHECK(world.objects[o].active == true);

    auto l = scope.alloc_light_slot();
    CHECK(world.lights[l].active == true);
}

TEST_CASE("active flag is false after free, true after re-alloc") {
    RenderWorld world;
    auto scope = world.begin_sync();

    auto o = scope.alloc_object_slot();
    scope.free_object_slot(o);
    CHECK(world.objects[o].active == false);

    auto o2 = scope.alloc_object_slot();
    CHECK(o2 == o);
    CHECK(world.objects[o2].active == true);

    auto l = scope.alloc_light_slot();
    scope.free_light_slot(l);
    CHECK(world.lights[l].active == false);

    auto l2 = scope.alloc_light_slot();
    CHECK(l2 == l);
    CHECK(world.lights[l2].active == true);
}

TEST_CASE("SyncScope bumps mesh_version once") {
    RenderWorld world;
    auto initial = world.mesh_version;
    {
        auto scope = world.begin_sync();
        scope.alloc_object_slot();
        scope.alloc_object_slot();
        scope.alloc_mesh_slot();
    }
    CHECK(world.mesh_version == initial + 1);
}
