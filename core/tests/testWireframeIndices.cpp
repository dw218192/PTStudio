#include <core/rendering/wireframeIndices.h>

#include "testApplication.h"

using namespace pts::rendering;

TEST_CASE("expand_wireframe_indices - single triangle") {
    uint32_t tri[] = {0, 1, 2};
    auto lines = expand_wireframe_indices(tri, 3);
    REQUIRE(lines.size() == 6);
    CHECK(lines[0] == 0);
    CHECK(lines[1] == 1);
    CHECK(lines[2] == 1);
    CHECK(lines[3] == 2);
    CHECK(lines[4] == 2);
    CHECK(lines[5] == 0);
}

TEST_CASE("expand_wireframe_indices - two triangles") {
    uint32_t tri[] = {0, 1, 2, 3, 4, 5};
    auto lines = expand_wireframe_indices(tri, 6);
    REQUIRE(lines.size() == 12);
    // First triangle edges
    CHECK(lines[0] == 0);
    CHECK(lines[1] == 1);
    CHECK(lines[2] == 1);
    CHECK(lines[3] == 2);
    CHECK(lines[4] == 2);
    CHECK(lines[5] == 0);
    // Second triangle edges
    CHECK(lines[6] == 3);
    CHECK(lines[7] == 4);
    CHECK(lines[8] == 4);
    CHECK(lines[9] == 5);
    CHECK(lines[10] == 5);
    CHECK(lines[11] == 3);
}

TEST_CASE("expand_wireframe_indices - empty input") {
    auto lines = expand_wireframe_indices(nullptr, 0);
    CHECK(lines.empty());
}

TEST_CASE("expand_wireframe_indices - shared vertices produce duplicate edges") {
    // Two triangles sharing an edge (0-1)
    uint32_t tri[] = {0, 1, 2, 0, 1, 3};
    auto lines = expand_wireframe_indices(tri, 6);
    REQUIRE(lines.size() == 12);
    // Edge 0-1 appears twice (once per triangle)
    CHECK(lines[0] == 0);
    CHECK(lines[1] == 1);
    CHECK(lines[6] == 0);
    CHECK(lines[7] == 1);
}

PTS_TEST_MAIN()
