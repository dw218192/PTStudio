#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/rendering/renderWorld.h>
#include <core/rendering/shadowLightProjection.h>
#include <doctest/doctest.h>

#include <glm/gtc/matrix_transform.hpp>

using namespace pts::rendering;

TEST_CASE("linearize_shadow_depth round-trips ortho NDC") {
    // glm::ortho (ZO): ndc = (view_dist - near) / (far - near).
    float np = 2.0f;
    float fp = 20.0f;
    CHECK(linearize_shadow_depth(0.0f, np, fp, 0) == doctest::Approx(np));
    CHECK(linearize_shadow_depth(1.0f, np, fp, 0) == doctest::Approx(fp));
    CHECK(linearize_shadow_depth(0.5f, np, fp, 0) == doctest::Approx(np + 0.5f * (fp - np)));
}

TEST_CASE("linearize_shadow_depth round-trips perspective NDC") {
    // Reconstruct a set of known view-space depths through the perspective
    // projection and verify we get them back from linearize_shadow_depth.
    float np = 0.1f;
    float fp = 100.0f;
    glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f, np, fp);

    for (float dist : {0.5f, 1.0f, 10.0f, 50.0f, 99.0f}) {
        glm::vec4 view_pos(0.0f, 0.0f, -dist, 1.0f);
        glm::vec4 clip = proj * view_pos;
        float ndc_z = clip.z / clip.w;
        float reconstructed = linearize_shadow_depth(ndc_z, np, fp, 1);
        CHECK(reconstructed == doctest::Approx(dist).epsilon(1e-4));
    }

    CHECK(linearize_shadow_depth(0.0f, np, fp, 1) == doctest::Approx(np));
    CHECK(linearize_shadow_depth(1.0f, np, fp, 1) == doctest::Approx(fp).epsilon(1e-4));
}

TEST_CASE("ShadowInfo matches the 96-byte GPU layout") {
    // The shader's ShadowInfo must stay in sync with C++ ShadowInfo.
    // static_assert in the header enforces total size; here we verify field
    // offsets so a reorder in either language is caught.
    CHECK(offsetof(ShadowInfo, light_vp) == 0u);
    CHECK(offsetof(ShadowInfo, texel_size) == 64u);
    CHECK(offsetof(ShadowInfo, normal_bias) == 68u);
    CHECK(offsetof(ShadowInfo, has_shadow) == 72u);
    CHECK(offsetof(ShadowInfo, layer) == 76u);
    CHECK(offsetof(ShadowInfo, light_near) == 80u);
    CHECK(offsetof(ShadowInfo, light_far) == 84u);
    CHECK(offsetof(ShadowInfo, light_size_uv) == 88u);
    CHECK(offsetof(ShadowInfo, projection_type) == 92u);
}
