#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "transformDecompose.h"

using pts::editor::compose_trs;
using pts::editor::decompose_trs;
using pts::editor::TransformComponents;

static bool mat4_approx_equal(const glm::mat4& a, const glm::mat4& b, float eps = 1e-4f) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (std::abs(a[i][j] - b[i][j]) > eps) return false;
    return true;
}

static bool vec3_approx_equal(const glm::vec3& a, const glm::vec3& b, float eps = 1e-4f) {
    return std::abs(a.x - b.x) < eps && std::abs(a.y - b.y) < eps && std::abs(a.z - b.z) < eps;
}

TEST_CASE("identity matrix decomposes to default TRS") {
    auto trs = decompose_trs(glm::mat4(1.f));
    CHECK(vec3_approx_equal(trs.translate, {0, 0, 0}));
    CHECK(vec3_approx_equal(trs.rotate_degrees, {0, 0, 0}));
    CHECK(vec3_approx_equal(trs.scale, {1, 1, 1}));
}

TEST_CASE("translation-only matrix") {
    glm::mat4 m = glm::translate(glm::mat4(1.f), {3.f, -2.f, 7.f});
    auto trs = decompose_trs(m);
    CHECK(vec3_approx_equal(trs.translate, {3, -2, 7}));
    CHECK(vec3_approx_equal(trs.rotate_degrees, {0, 0, 0}));
    CHECK(vec3_approx_equal(trs.scale, {1, 1, 1}));
}

TEST_CASE("scale-only matrix") {
    glm::mat4 m = glm::scale(glm::mat4(1.f), {2.f, 0.5f, 3.f});
    auto trs = decompose_trs(m);
    CHECK(vec3_approx_equal(trs.translate, {0, 0, 0}));
    CHECK(vec3_approx_equal(trs.rotate_degrees, {0, 0, 0}));
    CHECK(vec3_approx_equal(trs.scale, {2, 0.5f, 3}));
}

TEST_CASE("rotation-only matrix (45 deg around Y)") {
    float angle = glm::radians(45.f);
    glm::mat4 m = glm::eulerAngleXYZ(0.f, angle, 0.f);
    auto trs = decompose_trs(m);
    CHECK(vec3_approx_equal(trs.translate, {0, 0, 0}));
    CHECK(vec3_approx_equal(trs.scale, {1, 1, 1}));
    CHECK(trs.rotate_degrees.y == doctest::Approx(45.f).epsilon(0.01));
}

TEST_CASE("compose_trs inverts decompose_trs (roundtrip)") {
    TransformComponents c;
    c.translate = {1.f, -3.f, 5.f};
    c.rotate_degrees = {30.f, 45.f, -20.f};
    c.scale = {1.5f, 2.f, 0.8f};

    glm::mat4 m = compose_trs(c);
    auto trs = decompose_trs(m);

    CHECK(vec3_approx_equal(trs.translate, c.translate));
    CHECK(vec3_approx_equal(trs.rotate_degrees, c.rotate_degrees, 0.01f));
    CHECK(vec3_approx_equal(trs.scale, c.scale));
}

TEST_CASE("decompose then recompose produces the same matrix") {
    // Build an arbitrary TRS matrix
    glm::mat4 original =
        glm::translate(glm::mat4(1.f), {-4.f, 2.f, 1.f}) *
        glm::eulerAngleXYZ(glm::radians(15.f), glm::radians(-30.f), glm::radians(60.f)) *
        glm::scale(glm::mat4(1.f), {1.2f, 0.9f, 2.5f});

    auto trs = decompose_trs(original);
    glm::mat4 reconstructed = compose_trs(trs);
    CHECK(mat4_approx_equal(original, reconstructed));
}

TEST_CASE("uniform scale roundtrip") {
    TransformComponents c;
    c.scale = {3.f, 3.f, 3.f};
    c.translate = {0.f, 5.f, 0.f};

    glm::mat4 m = compose_trs(c);
    auto trs = decompose_trs(m);

    CHECK(vec3_approx_equal(trs.scale, {3, 3, 3}));
    CHECK(vec3_approx_equal(trs.translate, {0, 5, 0}));
}
