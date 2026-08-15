#include <core/rendering/bvh.h>
#include <core/rendering/packedTriangle.h>
#include <core/rendering/renderWorld.h>

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

// --- transform_aabb tests ---

TEST_CASE("transform_aabb - identity") {
    auto aabb = AABB::from_min_max({-1, -2, -3}, {4, 5, 6});
    auto result = transform_aabb(aabb, glm::mat4(1.0f));
    CHECK(result.min.x == doctest::Approx(-1));
    CHECK(result.min.y == doctest::Approx(-2));
    CHECK(result.min.z == doctest::Approx(-3));
    CHECK(result.max.x == doctest::Approx(4));
    CHECK(result.max.y == doctest::Approx(5));
    CHECK(result.max.z == doctest::Approx(6));
}

TEST_CASE("transform_aabb - translation") {
    auto aabb = AABB::from_min_max({0, 0, 0}, {1, 1, 1});
    glm::mat4 m(1.0f);
    m[3] = glm::vec4(10, 20, 30, 1);
    auto result = transform_aabb(aabb, m);
    CHECK(result.min.x == doctest::Approx(10));
    CHECK(result.min.y == doctest::Approx(20));
    CHECK(result.min.z == doctest::Approx(30));
    CHECK(result.max.x == doctest::Approx(11));
    CHECK(result.max.y == doctest::Approx(21));
    CHECK(result.max.z == doctest::Approx(31));
}

TEST_CASE("transform_aabb - uniform scale") {
    auto aabb = AABB::from_min_max({-1, -1, -1}, {1, 1, 1});
    glm::mat4 m(1.0f);
    m[0][0] = 3.0f;
    m[1][1] = 3.0f;
    m[2][2] = 3.0f;
    auto result = transform_aabb(aabb, m);
    CHECK(result.min.x == doctest::Approx(-3));
    CHECK(result.max.x == doctest::Approx(3));
    CHECK(result.min.y == doctest::Approx(-3));
    CHECK(result.max.y == doctest::Approx(3));
}

TEST_CASE("transform_aabb - 90-degree rotation around Z") {
    // After rotating 90 deg around Z: X->Y, Y->-X
    auto aabb = AABB::from_min_max({1, 0, 0}, {3, 1, 1});
    glm::mat4 m(1.0f);
    // column 0 = (0, 1, 0), column 1 = (-1, 0, 0)
    m[0] = glm::vec4(0, 1, 0, 0);
    m[1] = glm::vec4(-1, 0, 0, 0);
    m[2] = glm::vec4(0, 0, 1, 0);
    auto result = transform_aabb(aabb, m);
    // Center of input = (2, 0.5, 0.5), extent = (1, 0.5, 0.5)
    // Rotated center = (-0.5, 2, 0.5)
    // new_extent[0] = |m[0][0]|*1 + |m[1][0]|*0.5 + |m[2][0]|*0.5 = 0+1*0.5+0 = 0.5
    // new_extent[1] = |m[0][1]|*1 + |m[1][1]|*0.5 + |m[2][1]|*0.5 = 1+0+0 = 1
    CHECK(result.min.x == doctest::Approx(-1));
    CHECK(result.max.x == doctest::Approx(0));
    CHECK(result.min.y == doctest::Approx(1));
    CHECK(result.max.y == doctest::Approx(3));
}

// --- Two-level BVH concatenation test ---

TEST_CASE("BVH - TLAS over two BLAS produces valid concatenated tree") {
    // Build two BLAS in local space
    // Mesh A: triangles at x=[0..5]
    std::vector<AABB> mesh_a_aabbs;
    for (int i = 0; i < 5; ++i) {
        float x = static_cast<float>(i);
        mesh_a_aabbs.push_back(AABB::from_min_max({x, 0, 0}, {x + 1, 1, 0}));
    }
    BVH blas_a;
    blas_a.build(mesh_a_aabbs, 5);

    // Mesh B: triangles at x=[100..105]
    std::vector<AABB> mesh_b_aabbs;
    for (int i = 0; i < 5; ++i) {
        float x = 100.0f + static_cast<float>(i);
        mesh_b_aabbs.push_back(AABB::from_min_max({x, 0, 0}, {x + 1, 1, 0}));
    }
    BVH blas_b;
    blas_b.build(mesh_b_aabbs, 5);

    // Build TLAS over 2 instances
    AABB inst_a_aabb = AABB::from_min_max({0, 0, 0}, {5, 1, 0});
    AABB inst_b_aabb = AABB::from_min_max({100, 0, 0}, {105, 1, 0});
    std::vector<AABB> tlas_aabbs = {inst_a_aabb, inst_b_aabb};
    BVH tlas;
    tlas.build(tlas_aabbs, 2);

    uint32_t tlas_nc = tlas.node_count();

    // Concatenate TLAS + BLAS nodes
    std::vector<BVHNode> all_nodes;
    auto tlas_nodes = tlas.nodes();
    all_nodes.insert(all_nodes.end(), tlas_nodes.begin(), tlas_nodes.end());

    uint32_t blas_a_base = tlas_nc;
    for (const auto& node : blas_a.nodes()) {
        BVHNode n = node;
        if (n.count == 0) n.left_first += blas_a_base;
        all_nodes.push_back(n);
    }

    uint32_t blas_b_base = tlas_nc + blas_a.node_count();
    for (const auto& node : blas_b.nodes()) {
        BVHNode n = node;
        if (n.count == 0) n.left_first += blas_b_base;
        all_nodes.push_back(n);
    }

    // TLAS root should cover both instances
    CHECK(all_nodes[0].aabb_min.x == doctest::Approx(0));
    CHECK(all_nodes[0].aabb_max.x == doctest::Approx(105));

    // Verify all interior nodes point to valid children
    for (uint32_t i = 0; i < static_cast<uint32_t>(all_nodes.size()); ++i) {
        const auto& n = all_nodes[i];
        if (n.count == 0) {
            CHECK(n.left_first < static_cast<uint32_t>(all_nodes.size()));
            CHECK(n.left_first + 1 < static_cast<uint32_t>(all_nodes.size()));
        }
    }

    // Total nodes = TLAS + BLAS_A + BLAS_B
    CHECK(all_nodes.size() == tlas_nc + blas_a.node_count() + blas_b.node_count());
}

// --- build_from_mesh tests ---

TEST_CASE("BVH::build_from_mesh - builds BVH and returns reordered PackedTriangles") {
    // Two triangles forming a quad
    std::vector<Vertex> verts = {
        {{0, 0, 0}, {0, 0, 1}, {1, 1, 1}, {0, 0}},
        {{1, 0, 0}, {0, 0, 1}, {1, 1, 1}, {1, 0}},
        {{1, 1, 0}, {0, 0, 1}, {1, 1, 1}, {1, 1}},
        {{0, 1, 0}, {0, 0, 1}, {1, 1, 1}, {0, 1}},
    };
    std::vector<uint32_t> indices = {0, 1, 2, 0, 2, 3};

    BVH bvh;
    auto tris = bvh.build_from_mesh(verts, indices);

    CHECK(tris.size() == 2);
    CHECK(bvh.node_count() >= 1);
    CHECK(bvh.tri_indices().size() == 2);

    // Verify positions are populated correctly (in some reordered order)
    for (const auto& tri : tris) {
        // Each vertex position should be within [0,1]^2 x 0
        for (auto* v : {&tri.v0, &tri.v1, &tri.v2}) {
            CHECK(v->x >= doctest::Approx(0));
            CHECK(v->x <= doctest::Approx(1));
            CHECK(v->y >= doctest::Approx(0));
            CHECK(v->y <= doctest::Approx(1));
            CHECK(v->z == doctest::Approx(0));
        }
        // Normals should be (0, 0, 1)
        for (auto* n : {&tri.n0, &tri.n1, &tri.n2}) {
            CHECK(n->z == doctest::Approx(1));
        }
    }
}

TEST_CASE("BVH::build_from_mesh - many triangles produces valid tree") {
    // Build a strip of 20 triangles spread along X
    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
    for (uint32_t i = 0; i < 20; ++i) {
        float x = static_cast<float>(i) * 10.0f;
        uint32_t base = static_cast<uint32_t>(verts.size());
        verts.push_back({{x, 0, 0}, {0, 0, 1}, {}, {0, 0}});
        verts.push_back({{x + 1, 0, 0}, {0, 0, 1}, {}, {1, 0}});
        verts.push_back({{x, 1, 0}, {0, 0, 1}, {}, {0, 1}});
        indices.push_back(base);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
    }

    BVH bvh;
    auto tris = bvh.build_from_mesh(verts, indices);

    CHECK(tris.size() == 20);
    check_tree_validity(bvh, 20);
}

// --- concatenate_nodes tests ---

TEST_CASE("BVH::concatenate_nodes - matches manual concatenation") {
    // Build two BLAS
    std::vector<AABB> mesh_a_aabbs;
    for (int i = 0; i < 5; ++i) {
        float x = static_cast<float>(i);
        mesh_a_aabbs.push_back(AABB::from_min_max({x, 0, 0}, {x + 1, 1, 0}));
    }
    BVH blas_a;
    blas_a.build(mesh_a_aabbs, 5);

    std::vector<AABB> mesh_b_aabbs;
    for (int i = 0; i < 5; ++i) {
        float x = 100.0f + static_cast<float>(i);
        mesh_b_aabbs.push_back(AABB::from_min_max({x, 0, 0}, {x + 1, 1, 0}));
    }
    BVH blas_b;
    blas_b.build(mesh_b_aabbs, 5);

    // Build TLAS
    std::vector<AABB> tlas_aabbs = {
        AABB::from_min_max({0, 0, 0}, {5, 1, 0}),
        AABB::from_min_max({100, 0, 0}, {105, 1, 0}),
    };
    BVH tlas;
    tlas.build(tlas_aabbs, 2);

    uint32_t tlas_nc = tlas.node_count();

    // Use concatenate_nodes()
    std::vector<BlasEntry> entries = {
        {&blas_a, 0},
        {&blas_b, 5},
    };
    auto all_nodes = tlas.concatenate_nodes(entries);

    // Total nodes = TLAS + BLAS_A + BLAS_B
    CHECK(all_nodes.size() == tlas_nc + blas_a.node_count() + blas_b.node_count());

    // TLAS root covers both instances
    CHECK(all_nodes[0].aabb_min.x == doctest::Approx(0));
    CHECK(all_nodes[0].aabb_max.x == doctest::Approx(105));

    // All interior nodes point to valid children
    for (uint32_t i = 0; i < static_cast<uint32_t>(all_nodes.size()); ++i) {
        const auto& n = all_nodes[i];
        if (n.count == 0) {
            CHECK(n.left_first < static_cast<uint32_t>(all_nodes.size()));
            CHECK(n.left_first + 1 < static_cast<uint32_t>(all_nodes.size()));
        }
    }
}

TEST_CASE("BVH::concatenate_nodes - empty blas_list returns TLAS nodes only") {
    std::vector<AABB> tlas_aabbs = {AABB::from_min_max({0, 0, 0}, {1, 1, 1})};
    BVH tlas;
    tlas.build(tlas_aabbs, 1);

    auto all_nodes = tlas.concatenate_nodes({});

    CHECK(all_nodes.size() == tlas.node_count());
    CHECK(all_nodes[0].aabb_min == tlas.nodes()[0].aabb_min);
    CHECK(all_nodes[0].aabb_max == tlas.nodes()[0].aabb_max);
}

PTS_TEST_MAIN()
