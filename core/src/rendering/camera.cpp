#include <core/rendering/camera.h>

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace pts::rendering {

OrbitCamera::OrbitCamera() = default;

void OrbitCamera::set_target(glm::vec3 target) {
    m_target = target;
}

void OrbitCamera::set_distance(float distance) {
    m_distance = std::clamp(distance, m_min_distance, m_max_distance);
}

void OrbitCamera::set_fov_y(float fov_degrees) {
    m_fov_y = fov_degrees;
}

void OrbitCamera::set_yaw(float radians) {
    m_yaw = radians;
}

void OrbitCamera::set_pitch(float radians) {
    m_pitch = std::clamp(radians, -k_max_pitch, k_max_pitch);
}

void OrbitCamera::set_clip_planes(float near, float far) {
    m_near = near;
    m_far = far;
}

void OrbitCamera::set_up_axis(UpAxis axis) {
    m_up_axis = axis;
}

void OrbitCamera::apply_meters_per_unit(float meters_per_unit) {
    float s = 1.0f / meters_per_unit;  // scene units per meter
    m_near = 0.01f * s;
    m_far = 2000.0f * s;
    m_min_distance = 0.01f * s;
    m_max_distance = 1000.0f * s;
    m_move_speed = 5.0f * s;
    m_distance = std::clamp(m_distance, m_min_distance, m_max_distance);
}

void OrbitCamera::orbit(float dx, float dy) {
    m_yaw -= dx * k_orbit_speed;
    m_pitch += dy * k_orbit_speed;
    m_pitch = std::clamp(m_pitch, -k_max_pitch, k_max_pitch);
}

void OrbitCamera::pan(float dx, float dy) {
    auto view = view_matrix();
    auto right = glm::vec3(view[0][0], view[1][0], view[2][0]);
    auto up = glm::vec3(view[0][1], view[1][1], view[2][1]);
    m_target += (-dx * right + dy * up) * m_distance * k_pan_speed;
}

void OrbitCamera::zoom(float delta) {
    m_distance *= (1.0f - delta * k_zoom_speed);
    m_distance = std::max(m_distance, m_near);
}

void OrbitCamera::move(float forward, float right_amount, float up_amount, float dt) {
    auto view = view_matrix();
    auto right_dir = glm::vec3(view[0][0], view[1][0], view[2][0]);
    auto up_dir =
        (m_up_axis == UpAxis::Z) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
    auto forward_dir = glm::normalize(m_target - position());

    auto offset =
        (forward_dir * forward + right_dir * right_amount + up_dir * up_amount) * m_move_speed * dt;
    m_target += offset;
}

auto OrbitCamera::view_matrix() const -> glm::mat4 {
    auto up = (m_up_axis == UpAxis::Z) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
    return glm::lookAt(position(), m_target, up);
}

auto OrbitCamera::projection_matrix(float aspect_ratio) const -> glm::mat4 {
    return glm::perspective(glm::radians(m_fov_y), aspect_ratio, m_near, m_far);
}

auto OrbitCamera::target() const -> glm::vec3 {
    return m_target;
}

auto OrbitCamera::position() const -> glm::vec3 {
    if (m_up_axis == UpAxis::Z) {
        // Z-up: pitch elevates from XY plane toward +Z
        float x = m_target.x + m_distance * std::cos(m_pitch) * std::cos(m_yaw);
        float y = m_target.y + m_distance * std::cos(m_pitch) * std::sin(m_yaw);
        float z = m_target.z + m_distance * std::sin(m_pitch);
        return {x, y, z};
    }
    // Y-up: pitch elevates from XZ plane toward +Y
    float x = m_target.x + m_distance * std::cos(m_pitch) * std::sin(m_yaw);
    float y = m_target.y + m_distance * std::sin(m_pitch);
    float z = m_target.z + m_distance * std::cos(m_pitch) * std::cos(m_yaw);
    return {x, y, z};
}

auto OrbitCamera::distance() const -> float {
    return m_distance;
}

auto OrbitCamera::fov_y_degrees() const -> float {
    return m_fov_y;
}

auto OrbitCamera::near_plane() const -> float {
    return m_near;
}

auto OrbitCamera::far_plane() const -> float {
    return m_far;
}

auto OrbitCamera::up_axis() const -> UpAxis {
    return m_up_axis;
}

}  // namespace pts::rendering
