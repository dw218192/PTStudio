#include <core/rendering/bvh.h>

#include <algorithm>
#include <numeric>
#include <set>

#include "testApplication.h"

using namespace pts::rendering;

namespace {

std::vector<AABB> make_tri_aabbs(const std::vector<std::array<glm::vec3, 3>>& tris) {
    std::vector<AABB> aabbs;
    aabbs.reserve(tris.size());
    for (auto& t : tris) {
        AABB a = AABB::from_point(t[0]);
        a.expand(t[1]);
        a.expand(t[2]);
        aabbs.push_back(a);
    }
    return aabbs;
}

void check_tree_validity(const BVH& bvh, uint32_t tri_count) {
    for (uint32_t i = 0; i < bvh.node_count(); ++i) {
        auto& n = bvh.nodes()[i];
        if (n.count == 0) {
            CHECK(n.left_first + 1 < bvh.node_count());
        } else {
            CHECK(n.left_first + n.count <= tri_count);
        }
    }
}

}  // namespace

TEST_CASE("BVH - empty input") {
    BVH bvh;
    bvh.build({}, 0);
    REQUIRE(bvh.node_count() == 1);
    CHECK(bvh.tri_indices().empty());
}

TEST_CASE("BVH - single triangle") {
    std::vector<std::array<glm::vec3, 3>> tris = {
        {glm::vec3{0, 0, 0}, glm::vec3{1, 0, 0}, glm::vec3{0, 1, 0}}};
    auto aabbs = make_tri_aabbs(tris);

    BVH bvh;
    bvh.build(aabbs, 1);

    REQUIRE(bvh.node_count() == 1);
    CHECK(bvh.nodes()[0].count == 1);
    REQUIRE(bvh.tri_indices().size() == 1);
    CHECK(bvh.tri_indices()[0] == 0);

    CHECK(bvh.scene_bounds().min == glm::vec3(0, 0, 0));
    CHECK(bvh.scene_bounds().max == glm::vec3(1, 1, 0));
}

TEST_CASE("BVH - two clusters far apart") {
    std::vector<std::array<glm::vec3, 3>> tris;
    for (int i = 0; i < 5; ++i) {
        float y = static_cast<float>(i);
        tris.push_back({glm::vec3{0, y, 0}, glm::vec3{1, y, 0}, glm::vec3{0, y + 1, 0}});
    }
    for (int i = 0; i < 5; ++i) {
        float y = static_cast<float>(i);
        tris.push_back({glm::vec3{100, y, 0}, glm::vec3{101, y, 0}, glm::vec3{100, y + 1, 0}});
    }
    auto aabbs = make_tri_aabbs(tris);

    BVH bvh;
    bvh.build(aabbs, 10);

    CHECK(bvh.nodes()[0].count == 0);  // root is interior (split occurred)
    uint32_t left = bvh.nodes()[0].left_first;
    REQUIRE(left + 1 < bvh.node_count());
    REQUIRE(bvh.tri_indices().size() == 10);
}

TEST_CASE("BVH - cube (12 triangles)") {
    std::vector<std::array<glm::vec3, 3>> tris;
    glm::vec3 v[8] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
    };
    auto face = [&](int a, int b, int c, int d) {
        tris.push_back({v[a], v[b], v[c]});
        tris.push_back({v[a], v[c], v[d]});
    };
    face(0, 1, 2, 3);
    face(5, 4, 7, 6);
    face(4, 0, 3, 7);
    face(1, 5, 6, 2);
    face(3, 2, 6, 7);
    face(4, 5, 1, 0);
    REQUIRE(tris.size() == 12);

    auto aabbs = make_tri_aabbs(tris);
    BVH bvh;
    bvh.build(aabbs, 12);

    REQUIRE(bvh.tri_indices().size() == 12);
    std::vector<uint32_t> sorted(bvh.tri_indices().begin(), bvh.tri_indices().end());
    std::sort(sorted.begin(), sorted.end());
    for (uint32_t i = 0; i < 12; ++i) {
        CHECK(sorted[i] == i);
    }

    CHECK(bvh.scene_bounds().min == glm::vec3(0, 0, 0));
    CHECK(bvh.scene_bounds().max == glm::vec3(1, 1, 1));
    check_tree_validity(bvh, 12);
}

TEST_CASE("BVH - degenerate (all same position)") {
    constexpr uint32_t n = 10;
    std::vector<std::array<glm::vec3, 3>> tris;
    for (uint32_t i = 0; i < n; ++i) {
        tris.push_back({glm::vec3{5, 5, 5}, glm::vec3{5, 5, 5}, glm::vec3{5, 5, 5}});
    }
    auto aabbs = make_tri_aabbs(tris);

    BVH bvh;
    bvh.build(aabbs, n);

    REQUIRE(bvh.tri_indices().size() == n);
    check_tree_validity(bvh, n);
}

TEST_CASE("BVH - tri_indices is a permutation") {
    constexpr uint32_t n = 64;
    std::vector<std::array<glm::vec3, 3>> tris;
    for (uint32_t i = 0; i < n; ++i) {
        float x = static_cast<float>(i) * 10.0f;
        tris.push_back({glm::vec3{x, 0, 0}, glm::vec3{x + 1, 0, 0}, glm::vec3{x, 1, 0}});
    }
    auto aabbs = make_tri_aabbs(tris);

    BVH bvh;
    bvh.build(aabbs, n);

    REQUIRE(bvh.tri_indices().size() == n);
    std::set<uint32_t> seen(bvh.tri_indices().begin(), bvh.tri_indices().end());
    CHECK(seen.size() == n);
    CHECK(*seen.begin() == 0);
    CHECK(*seen.rbegin() == n - 1);
}

TEST_CASE("BVH - node validity") {
    constexpr uint32_t n = 32;
    std::vector<std::array<glm::vec3, 3>> tris;
    for (uint32_t i = 0; i < n; ++i) {
        float x = static_cast<float>(i) * 5.0f;
        float y = static_cast<float>(i % 4) * 5.0f;
        tris.push_back({glm::vec3{x, y, 0}, glm::vec3{x + 1, y, 0}, glm::vec3{x, y + 1, 0}});
    }
    auto aabbs = make_tri_aabbs(tris);

    BVH bvh;
    bvh.build(aabbs, n);
    check_tree_validity(bvh, n);

    uint32_t total_leaf_tris = 0;
    for (auto& nd : bvh.nodes()) {
        if (nd.count > 0) {
            CHECK(nd.count <= k_bvh_max_leaf_size * 2);
            total_leaf_tris += nd.count;
        }
    }
    CHECK(total_leaf_tris == n);
}

TEST_CASE("BVH - scene AABB matches union of inputs") {
    std::vector<std::array<glm::vec3, 3>> tris = {
        {glm::vec3{-5, -3, -1}, glm::vec3{2, 0, 0}, glm::vec3{0, 4, 0}},
        {glm::vec3{0, 0, 0}, glm::vec3{10, 0, 0}, glm::vec3{0, 0, 7}},
    };
    auto aabbs = make_tri_aabbs(tris);

    BVH bvh;
    bvh.build(aabbs, 2);

    CHECK(bvh.scene_bounds().min == glm::vec3(-5, -3, -1));
    CHECK(bvh.scene_bounds().max == glm::vec3(10, 4, 7));
}

PTS_TEST_MAIN()
