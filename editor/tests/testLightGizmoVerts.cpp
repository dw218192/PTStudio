#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

// diagnostics.h defines CHECK which conflicts with doctest -- include it first
// then undef before doctest redefines it.
#include <core/diagnostics.h>
#undef CHECK

#include <doctest/doctest.h>

#include <cmath>
#include <glm/gtc/constants.hpp>

#include "passes/editorPass.h"

using pts::editor::generate_light_verts;
using pts::rendering::LightData;

TEST_CASE("Distant light generates circle + arrow vertices") {
    LightData light;
    light.type = LightData::Type::Distant;
    auto verts = generate_light_verts(light);

    // 48 circle segments * 2 + arrow shaft (2) + arrowhead (4 lines * 2) = 106
    REQUIRE(verts.size() == 106);
}

TEST_CASE("Distant light circle lies in XY plane") {
    LightData light;
    light.type = LightData::Type::Distant;
    auto verts = generate_light_verts(light);

    // First 96 vertices are the circle -- all Z should be 0
    for (size_t i = 0; i < 96; ++i) {
        CHECK(verts[i].z == doctest::Approx(0.0f));
    }
}

TEST_CASE("Distant light circle has correct radius") {
    LightData light;
    light.type = LightData::Type::Distant;
    auto verts = generate_light_verts(light);

    constexpr float expected_r = 0.5f;
    // Check that circle vertices are at the expected radius
    for (size_t i = 0; i < 96; ++i) {
        float r = std::sqrt(verts[i].x * verts[i].x + verts[i].y * verts[i].y);
        CHECK(r == doctest::Approx(expected_r).epsilon(0.001));
    }
}

TEST_CASE("Distant light arrow points along -Z") {
    LightData light;
    light.type = LightData::Type::Distant;
    auto verts = generate_light_verts(light);

    // Arrow shaft: verts[96] = origin, verts[97] = tip at -Z
    CHECK(verts[96].x == doctest::Approx(0.0f));
    CHECK(verts[96].y == doctest::Approx(0.0f));
    CHECK(verts[96].z == doctest::Approx(0.0f));
    CHECK(verts[97].x == doctest::Approx(0.0f));
    CHECK(verts[97].y == doctest::Approx(0.0f));
    CHECK(verts[97].z == doctest::Approx(-1.0f));
}

TEST_CASE("Distant light arrowhead lines originate from arrow tip") {
    LightData light;
    light.type = LightData::Type::Distant;
    auto verts = generate_light_verts(light);

    // 4 arrowhead lines: each starts from verts[98,100,102,104] at the tip
    for (size_t i = 98; i < 106; i += 2) {
        CHECK(verts[i].z == doctest::Approx(-1.0f));
    }
}

TEST_CASE("Sphere light generates 3 circles") {
    LightData light;
    light.type = LightData::Type::Sphere;
    light.radius = 1.0f;
    auto verts = generate_light_verts(light);
    CHECK(verts.size() == 48 * 2 * 3);
}

TEST_CASE("Dome light generates no vertices") {
    LightData light;
    light.type = LightData::Type::Dome;
    auto verts = generate_light_verts(light);
    CHECK(verts.empty());
}

TEST_CASE("Rect light generates rectangle + arrow") {
    LightData light;
    light.type = LightData::Type::Rect;
    light.width = 2.0f;
    light.height = 1.0f;
    auto verts = generate_light_verts(light);
    // 4 edges (8 verts) + arrow (2 verts) = 10
    CHECK(verts.size() == 10);
}

TEST_CASE("Disk light generates circle + arrow") {
    LightData light;
    light.type = LightData::Type::Disk;
    light.radius = 0.5f;
    auto verts = generate_light_verts(light);
    // 48 segments * 2 + arrow (2) = 98
    CHECK(verts.size() == 98);
}
