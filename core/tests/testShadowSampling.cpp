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

TEST_CASE("ShadowInfo matches the 144-byte GPU layout") {
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
    CHECK(offsetof(ShadowInfo, light_position) == 96u);
    CHECK(offsetof(ShadowInfo, light_u) == 112u);
    CHECK(offsetof(ShadowInfo, light_v) == 128u);
}

TEST_CASE("area shadow cube covers receivers when the emitter slides sideways") {
    LightData light;
    light.type = LightData::Type::Rect;
    const glm::vec3 bounds_min(-5, 0, -5), bounds_max(5, 3, 5);
    for (float light_x : {-12.0f, -5.0f, 0.0f, 5.0f, 12.0f}) {
        light.transform = glm::translate(glm::mat4(1), glm::vec3(light_x, 1.5f, 5));
        for (float x : {-5.0f, 0.0f, 5.0f}) {
            for (float z : {-5.0f, 0.0f, 5.0f}) {
                glm::vec3 receiver(x, 0, z);
                bool covered = false;
                for (uint32_t face = 0; face < 6; ++face) {
                    auto projection = compute_area_light_vp(light, bounds_min, bounds_max, face);
                    auto clip = projection.vp * glm::vec4(receiver, 1);
                    if (clip.w <= 0) continue;
                    auto ndc = glm::vec3(clip) / clip.w;
                    if (std::abs(ndc.x) <= 1.00001f && std::abs(ndc.y) <= 1.00001f && ndc.z >= 0 &&
                        ndc.z <= 1.00001f) {
                        covered = true;
                        CHECK(linearize_shadow_depth(ndc.z, projection.near_plane,
                                                     projection.far_plane,
                                                     2) == doctest::Approx(clip.w).epsilon(0.002));
                    }
                }
                CHECK(covered);
            }
        }
    }
}
