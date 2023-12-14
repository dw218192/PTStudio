#include "include/camera.h"

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_access.hpp>

static constexpr auto k_default_fov = 45.0f;
static constexpr auto k_default_aspect = 1.33f;
static GLM_CONSTEXPR auto k_default_look_at = PTS::LookAtParams {
    glm::vec3{ 0.0f, 0.0f, 5.0f },
    glm::vec3{ 0.0f, 0.0f, 0.0f },
    glm::vec3{ 0.0f, 1.0f, 0.0f }
};

PTS::Camera::Camera() noexcept : Camera{ k_default_fov, k_default_aspect, k_default_look_at } { }

PTS::Camera::Camera(float fovy, float aspect, LookAtParams const& params) noexcept
    : Camera{ fovy, aspect, params.eye, params.center, params.up } { }

PTS::Camera::Camera(float fovy, float aspect, glm::vec3 eye, glm::vec3 center, glm::vec3 up) noexcept
    : m_fov(fovy), m_aspect(aspect), m_eye{eye}, m_center{center}, m_up{up}
{
    on_deserialize();
}

auto PTS::Camera::ndc_to_wrold(glm::vec2 ndc, float z) const noexcept -> glm::vec3 {
    auto ret = m_inv_view_proj * glm::vec4{ ndc, z, 1.0f };
    ret /= ret.w;
    return ret;
}

auto PTS::Camera::world_to_ndc(glm::vec3 world) const noexcept -> glm::vec3 {
    auto ret = m_view_proj * glm::vec4(world, 1.0f);
    ret /= ret.w;
    return ret;
}

auto PTS::Camera::world_to_viewport(glm::vec3 world, glm::ivec2 vp_size) const noexcept -> glm::vec2 {
    auto ndc = world_to_ndc(world);
    // convert to (0, 1) and invert y
    ndc = (ndc + 1.0f) / 2.0f;
    ndc.y = 1 - ndc.y;
    return glm::vec2{
        ndc.x * vp_size.x, ndc.y * vp_size.y
    };
}

auto PTS::Camera::viewport_to_world(glm::vec2 screen, glm::ivec2 vp_size, float z) const noexcept -> glm::vec3 {
    auto ndc = glm::vec2{
        screen.x / vp_size.x, screen.y / vp_size.y
    };
    // invert y and convert to (-1, 1) 
    ndc.y = 1 - ndc.y;
    ndc = ndc * 2.0f - 1.0f;
    return ndc_to_wrold(ndc, z);
}

auto PTS::Camera::viewport_to_ray(glm::vec2 screen, glm::ivec2 vp_size) const noexcept -> Ray {
    auto world = viewport_to_world(screen, vp_size, 1.0f);
    return Ray{ m_eye, world - m_eye };
}

void PTS::Camera::set_delta_rotation(glm::vec3 const& delta) noexcept {
    auto rot_mat = glm::eulerAngleXYZ(glm::radians(delta.x), glm::radians(delta.y), glm::radians(delta.z));
    m_arm_dir = rot_mat * glm::vec4{ m_arm_dir, 1.0f };
    on_view_changed();
}

void PTS::Camera::set_delta_dolly(glm::vec3 const& delta) noexcept {
    glm::vec3 right = glm::column(m_cam_transform, 0);
    glm::vec3 up = glm::column(m_cam_transform, 1);
    glm::vec3 forward = glm::column(m_cam_transform, 2);
    auto world_delta = delta.x * right + delta.y * up + delta.z * forward;
    m_center += world_delta;
    on_view_changed();
}

void PTS::Camera::set_delta_zoom(float delta) noexcept {
    if (glm::sign(delta) > 0) {
        m_arm_dir *= 1.1f;
    } else {
        m_arm_dir *= 0.9f;
    }
    on_view_changed();
}

void PTS::Camera::set_eye(glm::vec3 const& eye) noexcept {
    m_eye = eye;
    on_deserialize();
}

void PTS::Camera::set_center(glm::vec3 const& center) noexcept {
    m_center = center;
    on_deserialize();
}

void PTS::Camera::set(LookAtParams const& params) noexcept {
    m_eye = params.eye;
    m_center = params.center;
    m_up = params.up;
    on_deserialize();
}

void PTS::Camera::on_deserialize() noexcept {
    m_arm_dir = m_eye - m_center;
    m_view = glm::lookAt(m_eye, m_center, m_up);
    m_cam_transform = glm::inverse(m_view);
    m_projection = glm::perspective(glm::radians(m_fov), m_aspect, k_near, k_far);
    m_view_proj = m_projection * m_view;
    m_inv_view_proj = glm::inverse(m_view_proj);
}

auto PTS::Camera::operator==(Camera const& other) const noexcept -> bool {
	return m_view_proj == other.m_view_proj;
}

auto PTS::Camera::operator!=(Camera const& other) const noexcept -> bool {
    return !(*this == other);
}

void PTS::Camera::set_fov(float fov) noexcept {
    m_fov = glm::clamp(fov, k_min_fov, k_max_fov);
    m_projection = glm::perspective(glm::radians(m_fov), m_aspect, k_near, k_far);
    on_proj_changed();
}

void PTS::Camera::set_aspect(float aspect) noexcept {
    m_aspect = aspect;
    m_projection = glm::perspective(glm::radians(m_fov), m_aspect, k_near, k_far);
    on_proj_changed();
}

void PTS::Camera::on_view_changed() noexcept {
    m_eye = m_center + m_arm_dir;
    m_view = glm::lookAt(m_eye, m_center, m_up);
    m_cam_transform = glm::inverse(m_view);
    on_proj_changed();
}

void PTS::Camera::on_proj_changed() noexcept {
    m_view_proj = m_projection * m_view;
    m_inv_view_proj = glm::inverse(m_view_proj);
}
