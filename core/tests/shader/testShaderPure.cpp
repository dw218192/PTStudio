#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/shader_pure.h>
#include <core/shader_types.h>
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>

TEST_CASE("Slang shadow penumbra respects geometry and emitter size") {
    CHECK(pts_pcss_penumbra_uv(10, 5, 0.2f, 0) == doctest::Approx(0.2f));
    CHECK(pts_pcss_penumbra_uv(10, 5, 0.2f, 1) == doctest::Approx(0.02f));
    CHECK(pts_pcss_penumbra_uv(5, 5, 0.2f, 0) == 0);
    CHECK(pts_pcss_penumbra_uv(4, 5, 0.2f, 0) == 0);
    CHECK(pts_pcss_penumbra_uv(10, 5, 0, 0) == 0);
    CHECK(std::isfinite(pts_pcss_penumbra_uv(10, 0, 0.2f, 0)));
}

TEST_CASE("Slang dielectric Fresnel satisfies optical boundary cases") {
    CHECK(pts_fresnel_dielectric(1, 1.0f / 1.5f) == doctest::Approx(0.04f));
    CHECK(pts_fresnel_dielectric(1, 1.5f) == doctest::Approx(0.04f));
    CHECK(pts_fresnel_dielectric(0.5f, 1.5f) == 1);  // total internal reflection
    CHECK(pts_fresnel_dielectric(0.5f, 1) == doctest::Approx(0));
    for (int i = 0; i <= 100; ++i) {
        const float f = pts_fresnel_dielectric(i / 100.0f, 1.0f / 1.5f);
        CHECK(std::isfinite(f));
        CHECK(f >= 0);
        CHECK(f <= 1);
    }
}

TEST_CASE("Slang temporal resolve retains or rejects history as requested") {
    CHECK(pts_temporal_ema_blend(0.8f, 0.2f, 1) == doctest::Approx(0.8f));
    CHECK(pts_temporal_ema_blend(0.8f, 0.2f, 0) == doctest::Approx(0.2f));
    CHECK(pts_temporal_ema_blend(0.8f, 0.2f, 0.25f) == doctest::Approx(0.35f));
    CHECK(pts_clamp_history(0.9f, 0.5f, 0.1f, 2) == doctest::Approx(0.7f));
    CHECK(pts_clamp_history(0.1f, 0.5f, 0.1f, 2) == doctest::Approx(0.3f));
    CHECK(pts_clamp_history(0.6f, 0.5f, 0.1f, 2) == doctest::Approx(0.6f));
    CHECK(pts_clamp_history(0.9f, 0.5f, 0, 2) == doctest::Approx(0.5f));
}

TEST_CASE("Slang tangent frame preserves vectors through local/world conversion") {
    float x = 0, y = 0, z = 0;
    pts_tangent_roundtrip(0.25f, -0.5f, 1, &x, &y, &z);
    CHECK(x == doctest::Approx(0.25f));
    CHECK(y == doctest::Approx(-0.5f));
    CHECK(z == doctest::Approx(1));
}

TEST_CASE("Slang BRDF samples are normalized and have finite nonnegative PDFs") {
    for (float roughness : {0.0f, 0.05f, 0.5f, 1.0f}) {
        for (uint32_t seed = 0; seed < 1024; ++seed) {
            float x = 0, y = 0, z = 0, pdf = 0;
            pts_sample_brdf(seed, roughness, &x, &y, &z, &pdf);
            CHECK(x * x + y * y + z * z == doctest::Approx(1).epsilon(4e-5));
            CHECK(std::isfinite(pdf));
            CHECK(pdf >= 0);
            if (z <= 0) CHECK(pdf == 0);
        }
    }
}

TEST_CASE("Generated shadow upload structs start inactive and preserve matrix storage") {
    using namespace pts::rendering;
    // The generated header checks every size and offset against Slang reflection.
    ShadowInfo shadow;
    CHECK(shadow.has_shadow == 0);
    CHECK(shadow.layer == 0);
    CHECK(shadow.projection_type == 0);
    ShadowVisibilityUniforms uniforms;
    CHECK(uniforms.frame_index == 0);
    // GLM assignment must retain column-major order for GPU upload.
    shadow.light_vp = glm::mat4(1);
    shadow.light_vp[3] = glm::vec4(2, 3, 4, 1);
    const auto* bytes = reinterpret_cast<const unsigned char*>(&shadow.light_vp);
    float translation[4];
    std::memcpy(translation, bytes + 3 * sizeof(shadow.light_vp[0]), sizeof(translation));
    CHECK(translation[0] == 2);
    CHECK(translation[1] == 3);
    CHECK(translation[2] == 4);
}
