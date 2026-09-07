#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/rendering/renderWorld.h>
#include <core/rendering/shadowMapPass.h>
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

TEST_CASE("area shadow cube covers receivers when the emitter slides sideways") {
    LightData light;
    light.type = LightData::Type::Rect;
    const glm::vec3 bounds_min(-5, 0, -5), bounds_max(5, 3, 5);
    for (float light_x : {-12.0f, -5.0f, 0.0f, 5.0f, 12.0f}) {
        light.transform = glm::translate(glm::mat4(1), glm::vec3(light_x, 1.5f, 5));
        auto projection = compute_shadow_projection(light, bounds_min, bounds_max);
        REQUIRE(projection.layer_count == 6);
        for (float x : {-5.0f, 0.0f, 5.0f}) {
            for (float z : {-5.0f, 0.0f, 5.0f}) {
                glm::vec3 receiver(x, 0, z);
                bool covered = false;
                for (uint32_t face = 0; face < projection.layer_count; ++face) {
                    auto clip = projection.layer_vps[face] * glm::vec4(receiver, 1);
                    if (clip.w <= 0) continue;
                    auto ndc = glm::vec3(clip) / clip.w;
                    if (std::abs(ndc.x) <= 1.00001f && std::abs(ndc.y) <= 1.00001f && ndc.z >= 0 &&
                        ndc.z <= 1.00001f) {
                        covered = true;
                        CHECK(linearize_shadow_depth(ndc.z, projection.info.light_near,
                                                     projection.info.light_far,
                                                     2) == doctest::Approx(clip.w).epsilon(0.002));
                    }
                }
                CHECK(covered);
            }
        }
    }
}
