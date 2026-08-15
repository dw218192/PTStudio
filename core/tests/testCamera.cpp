#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/rendering/camera.h>
#include <doctest/doctest.h>

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using pts::rendering::OrbitCamera;
using pts::rendering::UpAxis;

TEST_CASE("OrbitCamera defaults to Y-up") {
    OrbitCamera cam;
    CHECK(cam.up_axis() == UpAxis::Y);
}

TEST_CASE("OrbitCamera Y-up position is above XZ plane at default pitch") {
    OrbitCamera cam;
    cam.set_target({0, 0, 0});
    cam.set_distance(10.0f);
    cam.set_yaw(0.0f);
    cam.set_pitch(0.3f);

    auto pos = cam.position();
    // Y-up: pitch > 0 means positive Y
    CHECK(pos.y > 0.0f);
    // Distance from target should equal 10
    CHECK(glm::length(pos) == doctest::Approx(10.0f).epsilon(0.01));
}

TEST_CASE("OrbitCamera Z-up position is above XY plane at default pitch") {
    OrbitCamera cam;
    cam.set_up_axis(UpAxis::Z);
    cam.set_target({0, 0, 0});
    cam.set_distance(10.0f);
    cam.set_yaw(0.0f);
    cam.set_pitch(0.3f);

    auto pos = cam.position();
    // Z-up: pitch > 0 means positive Z
    CHECK(pos.z > 0.0f);
    CHECK(glm::length(pos) == doctest::Approx(10.0f).epsilon(0.01));
}

TEST_CASE("OrbitCamera Z-up view_matrix uses Z as up vector") {
    OrbitCamera cam;
    cam.set_up_axis(UpAxis::Z);
    cam.set_target({0, 0, 0});
    cam.set_distance(5.0f);
    cam.set_yaw(0.0f);
    cam.set_pitch(0.3f);

    auto view = cam.view_matrix();
    // The up vector in view space should be close to (0,0,1) in world space.
    // Extract the camera up direction from view matrix (column 1 transposed).
    glm::vec3 view_up(view[0][1], view[1][1], view[2][1]);
    // The view_up should have significant Z component
    CHECK(std::abs(view_up.z) > 0.5f);
}

TEST_CASE("OrbitCamera Y-up view_matrix uses Y as up vector") {
    OrbitCamera cam;
    cam.set_target({0, 0, 0});
    cam.set_distance(5.0f);
    cam.set_yaw(0.0f);
    cam.set_pitch(0.3f);

    auto view = cam.view_matrix();
    glm::vec3 view_up(view[0][1], view[1][1], view[2][1]);
    CHECK(std::abs(view_up.y) > 0.5f);
}

TEST_CASE("OrbitCamera Z-up: zero pitch places camera in XY plane") {
    OrbitCamera cam;
    cam.set_up_axis(UpAxis::Z);
    cam.set_target({0, 0, 0});
    cam.set_distance(5.0f);
    cam.set_yaw(0.0f);
    cam.set_pitch(0.0f);

    auto pos = cam.position();
    CHECK(pos.z == doctest::Approx(0.0f).epsilon(0.01));
    CHECK(glm::length(pos) == doctest::Approx(5.0f).epsilon(0.01));
}

TEST_CASE("OrbitCamera Y-up: zero pitch places camera in XZ plane") {
    OrbitCamera cam;
    cam.set_target({0, 0, 0});
    cam.set_distance(5.0f);
    cam.set_yaw(0.0f);
    cam.set_pitch(0.0f);

    auto pos = cam.position();
    CHECK(pos.y == doctest::Approx(0.0f).epsilon(0.01));
    CHECK(glm::length(pos) == doctest::Approx(5.0f).epsilon(0.01));
}
