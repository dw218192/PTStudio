#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/rendering/renderWorld.h>
#include <core/rendering/shadowLightProjection.h>
#include <doctest/doctest.h>

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace pts::rendering;

namespace {

constexpr float k_eps = 1e-4f;

}  // namespace

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

TEST_CASE("compute_distant_light_vp produces ortho projection with tan-angle light_size_uv") {
    // Scene AABB centered at origin, dimensions 10 x 10 x 10.
    glm::vec3 aabb_min(-5.0f);
    glm::vec3 aabb_max(5.0f);

    LightData light;
    light.type = LightData::Type::Distant;
    light.direction = glm::vec3(0.0f, -1.0f, 0.0f);
    // 10 degrees full-angle -> half-angle 5 deg
    light.angle = 10.0f;

    LightProjection proj = compute_distant_light_vp(light, aabb_min, aabb_max);

    CHECK(proj.projection_type == 0u);
    CHECK(proj.near_plane > 0.0f);
    CHECK(proj.far_plane > proj.near_plane);

    // The ortho_width equals the X extent of the AABB in light space. For a
    // straight-down light, light-space X axis aligns with world X, so the
    // light-space X extent matches the AABB X extent: 10.
    constexpr float k_expected_ortho_width = 10.0f;
    float expected = std::tan(glm::radians(5.0f)) / k_expected_ortho_width;
    CHECK(proj.light_size_uv == doctest::Approx(expected).epsilon(1e-4));
}

TEST_CASE("compute_distant_light_vp handles zero angle without NaN") {
    glm::vec3 aabb_min(-1.0f);
    glm::vec3 aabb_max(1.0f);

    LightData light;
    light.type = LightData::Type::Distant;
    light.direction = glm::vec3(0.0f, -1.0f, 0.0f);
    light.angle = 0.0f;

    LightProjection proj = compute_distant_light_vp(light, aabb_min, aabb_max);
    CHECK(proj.light_size_uv == doctest::Approx(0.0f));
    CHECK(proj.projection_type == 0u);
}

TEST_CASE("compute_area_light_vp produces perspective with radius-based light_size_uv") {
    glm::vec3 aabb_min(-2.0f, 0.0f, -2.0f);
    glm::vec3 aabb_max(2.0f, 4.0f, 2.0f);

    LightData light;
    light.type = LightData::Type::Disk;
    light.transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 6.0f, 0.0f)) *
                      glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1, 0, 0));
    light.radius = 1.5f;

    LightProjection proj = compute_area_light_vp(light, aabb_min, aabb_max);

    CHECK(proj.projection_type == 1u);
    CHECK(proj.near_plane > 0.0f);
    CHECK(proj.far_plane > proj.near_plane);

    // 90 deg FOV -> light_size_uv = radius / (2 * tan(45 deg)) = radius / 2.
    float expected = 1.5f / 2.0f;
    CHECK(proj.light_size_uv == doctest::Approx(expected).epsilon(1e-4));
}

TEST_CASE("compute_area_light_vp uses sqrt(half-extents) for rect lights") {
    glm::vec3 aabb_min(-2.0f, 0.0f, -2.0f);
    glm::vec3 aabb_max(2.0f, 4.0f, 2.0f);

    LightData light;
    light.type = LightData::Type::Rect;
    light.transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 6.0f, 0.0f)) *
                      glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1, 0, 0));
    light.width = 2.0f;
    light.height = 4.0f;

    LightProjection proj = compute_area_light_vp(light, aabb_min, aabb_max);

    // Effective radius = sqrt((w/2) * (h/2)) = sqrt(1 * 2) = sqrt(2).
    float effective_radius = std::sqrt(1.0f * 2.0f);
    float expected = effective_radius / 2.0f;
    CHECK(proj.light_size_uv == doctest::Approx(expected).epsilon(1e-4));
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

TEST_CASE("penumbra grows with receiver-blocker depth delta (ortho formula)") {
    // Replicates the shader's ortho penumbra formula:
    //   penumbra_uv = light_size_uv * (d_r - d_b) / d_b
    auto penumbra = [](float light_size_uv, float receiver, float blocker) {
        return light_size_uv * std::max(receiver - blocker, 0.0f) / std::max(blocker, 1e-6f);
    };

    float lsu = 0.01f;  // 1% UV per unit depth ratio
    float p_small = penumbra(lsu, 10.0f, 9.0f);
    float p_large = penumbra(lsu, 10.0f, 2.0f);
    CHECK(p_small < p_large);
    CHECK(p_small > 0.0f);

    // Sanity: equal depths -> zero penumbra (receiver on blocker).
    CHECK(penumbra(lsu, 5.0f, 5.0f) == doctest::Approx(0.0f));
}

TEST_CASE("penumbra grows with receiver-blocker depth delta (perspective formula)") {
    // Replicates the shader's perspective penumbra formula:
    //   penumbra_uv = light_size_uv * (d_r - d_b) / (d_b * d_r)
    auto penumbra = [](float light_size_uv, float receiver, float blocker) {
        float num = std::max(receiver - blocker, 0.0f);
        float denom = std::max(blocker, 1e-6f) * std::max(receiver, 1e-6f);
        return light_size_uv * num / denom;
    };

    float lsu = 0.5f;  // half-meter equivalent
    float p_small = penumbra(lsu, 10.0f, 9.0f);
    float p_large = penumbra(lsu, 10.0f, 2.0f);
    CHECK(p_small < p_large);
    CHECK(p_small > 0.0f);
}
