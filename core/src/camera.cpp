#include "include/camera.h"

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>

Camera::Camera(float fovy, unsigned px_width, unsigned px_height, glm::vec3 eye, glm::vec3 center, glm::vec3 up) noexcept
    : m_fov(fovy), m_aspect(px_width / static_cast<float>(px_height)),
		m_px_width(px_width), m_px_height(px_height), m_eye{eye}, m_center{center}, m_up{up}, m_arm_dir{eye - center}
{
    m_view = glm::lookAt(eye, center, up);
    m_cam_transform = glm::inverse(m_view);
    m_projection = glm::perspective(glm::radians(fovy), m_aspect, k_near, k_far);
    m_view_proj = m_projection * m_view;
    m_inv_view_proj = glm::inverse(m_view_proj);
}

Camera::Camera(float fovy, unsigned px_width, unsigned px_height, LookAtParams const& params) noexcept
    : Camera{ fovy, px_width, px_height,params.eye, params.center, params.up } { }

auto Camera::ndc_to_wrold(glm::vec2 ndc, float z) const noexcept -> glm::vec3 {
    auto ret = m_inv_view_proj * glm::vec4{ ndc, z, 1.0f };
    ret /= ret.w;
    return ret;
}

auto Camera::world_to_ndc(glm::vec3 world) const noexcept -> glm::vec3 {
    auto ret = m_view_proj * glm::vec4(world, 1.0f);
    ret /= ret.w;
    return ret;
}

auto Camera::world_to_viewport(glm::vec3 world) const noexcept -> glm::vec2 {
    auto ndc = world_to_ndc(world);
    // convert to (0, 1) and invert y
    ndc = (ndc + 1.0f) / 2.0f;
    ndc.y = 1 - ndc.y;
    return glm::vec2{
        ndc.x * m_px_width, ndc.y * m_px_height
    };
}

auto Camera::viewport_to_world(glm::vec2 screen, float z) const noexcept -> glm::vec3 {
    auto ndc = glm::vec2{
        screen.x / m_px_width, screen.y / m_px_height
    };
    // invert y and convert to (-1, 1) 
    ndc.y = 1 - ndc.y;
    ndc = ndc * 2.0f - 1.0f;
    return ndc_to_wrold(ndc, z);
}
auto Camera::viewport_to_ray(glm::vec2 screen) const noexcept -> Ray {
    auto world = viewport_to_world(screen, 1.0f);
    return Ray{ m_eye, world - m_eye };
}

void Camera::set_delta_rotation(glm::vec3 const& delta) noexcept {
    auto rot_mat = glm::eulerAngleXYZ(glm::radians(delta.x), glm::radians(delta.y), glm::radians(delta.z));
    m_arm_dir = rot_mat * glm::vec4{ m_arm_dir, 1.0f };
    on_view_changed();
}

void Camera::set_delta_dolly(glm::vec3 const& delta) noexcept {
    m_center += delta;
    on_view_changed();
}

void Camera::set_delta_zoom(float delta) noexcept {
    if (glm::sign(delta) > 0) {
        m_arm_dir *= 1.1f;
    } else {
        m_arm_dir *= 0.9f;
    }
    on_view_changed();
}

void Camera::set_fov(float fov) noexcept {
    m_fov = glm::clamp(fov, k_min_fov, k_max_fov);
    m_projection = glm::perspective(glm::radians(m_fov), m_aspect, k_near, k_far);
    on_proj_changed();
}

void Camera::set_viewport(unsigned px_width, unsigned px_height) noexcept {
    m_px_width = px_width;
    m_px_height = px_height;
    m_aspect = px_width / static_cast<float>(px_height);
    m_projection = glm::perspective(glm::radians(m_fov), m_aspect, k_near, k_far);
    on_proj_changed();
}

void Camera::on_view_changed() noexcept {
    m_eye = m_center + m_arm_dir;
    m_view = glm::lookAt(m_eye, m_center, m_up);
    m_cam_transform = glm::inverse(m_view);
    on_proj_changed();
}

void Camera::on_proj_changed() noexcept {
    m_view_proj = m_projection * m_view;
    m_inv_view_proj = glm::inverse(m_view_proj);
}
