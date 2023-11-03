#include "include/camera.h"

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>

static constexpr auto k_default_fov = 45.0f;
static constexpr auto k_default_aspect = 1.33f;
static constexpr auto k_default_look_at = LookAtParams {
    glm::vec3{ 0.0f, 0.0f, 5.0f },
    glm::vec3{ 0.0f, 0.0f, 0.0f },
    glm::vec3{ 0.0f, 1.0f, 0.0f }
};

Camera::Camera() noexcept : Camera{ k_default_fov, k_default_aspect, k_default_look_at } { }
Camera::Camera(float fovy, float aspect, LookAtParams const& params) noexcept
    : Camera{ fovy, aspect, params.eye, params.center, params.up } { }

Camera::Camera(float fovy, float aspect, glm::vec3 eye, glm::vec3 center, glm::vec3 up) noexcept
    : m_fov(fovy), m_aspect(aspect), m_eye{eye}, m_center{center}, m_up{up}
{
    on_deserialize();
}

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

auto Camera::world_to_viewport(glm::vec3 world, glm::ivec2 vp_size) const noexcept -> glm::vec2 {
    auto ndc = world_to_ndc(world);
    // convert to (0, 1) and invert y
    ndc = (ndc + 1.0f) / 2.0f;
    ndc.y = 1 - ndc.y;
    return glm::vec2{
        ndc.x * vp_size.x, ndc.y * vp_size.y
    };
}

auto Camera::viewport_to_world(glm::vec2 screen, glm::ivec2 vp_size, float z) const noexcept -> glm::vec3 {
    auto ndc = glm::vec2{
        screen.x / vp_size.x, screen.y / vp_size.y
    };
    // invert y and convert to (-1, 1) 
    ndc.y = 1 - ndc.y;
    ndc = ndc * 2.0f - 1.0f;
    return ndc_to_wrold(ndc, z);
}
auto Camera::viewport_to_ray(glm::vec2 screen, glm::ivec2 vp_size) const noexcept -> Ray {
    auto world = viewport_to_world(screen, vp_size, 1.0f);
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

void Camera::on_deserialize() noexcept {
    m_arm_dir = m_eye - m_center;
    m_view = glm::lookAt(m_eye, m_center, m_up);
    m_cam_transform = glm::inverse(m_view);
    m_projection = glm::perspective(glm::radians(m_fov), m_aspect, k_near, k_far);
    m_view_proj = m_projection * m_view;
    m_inv_view_proj = glm::inverse(m_view_proj);
}

void Camera::set_fov(float fov) noexcept {
    m_fov = glm::clamp(fov, k_min_fov, k_max_fov);
    m_projection = glm::perspective(glm::radians(m_fov), m_aspect, k_near, k_far);
    on_proj_changed();
}

void Camera::set_aspect(float aspect) noexcept {
    m_aspect = aspect;
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
