#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

// diagnostics.h defines CHECK which conflicts with doctest -- include it first
// then undef before doctest redefines it.
#include <core/diagnostics.h>
#undef CHECK

#include <doctest/doctest.h>

#include "passes/editorPass.h"

using pts::editor::gizmo_distance_scale;

TEST_CASE("gizmo_distance_scale never shrinks below 1") {
    // Close camera -> scale stays at 1
    CHECK(gizmo_distance_scale(1.0f, 5.0f) == doctest::Approx(1.0f));
    CHECK(gizmo_distance_scale(0.0f, 1.0f) == doctest::Approx(1.0f));
}

TEST_CASE("gizmo_distance_scale grows with distance") {
    float s1 = gizmo_distance_scale(10.0f, 0.1f);
    float s2 = gizmo_distance_scale(100.0f, 0.1f);
    CHECK(s2 > s1);
    CHECK(s1 > 1.0f);
}

TEST_CASE("gizmo_distance_scale clamps tiny world_radius to 0.1") {
    // world_radius=0 should behave like 0.1
    float with_zero = gizmo_distance_scale(10.0f, 0.0f);
    float with_min = gizmo_distance_scale(10.0f, 0.1f);
    CHECK(with_zero == doctest::Approx(with_min));
}

TEST_CASE("gizmo_distance_scale respects custom min_screen_radius") {
    float small_screen = gizmo_distance_scale(20.0f, 0.5f, 0.02f);
    float large_screen = gizmo_distance_scale(20.0f, 0.5f, 0.10f);
    CHECK(large_screen > small_screen);
}

TEST_CASE("larger world radius needs less scaling") {
    // At large distance both exceed 1.0, but larger radius needs less scale
    float large_radius = gizmo_distance_scale(200.0f, 2.0f);
    float small_radius = gizmo_distance_scale(200.0f, 0.5f);
    CHECK(small_radius > large_radius);
    CHECK(large_radius > 1.0f);
}
